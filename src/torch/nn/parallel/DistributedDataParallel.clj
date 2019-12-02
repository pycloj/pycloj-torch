(ns torch.nn.parallel.DistributedDataParallel
  "Implements distributed data parallelism that is based on
    ``torch.distributed`` package at the module level.

    This container parallelizes the application of the given module by
    splitting the input across the specified devices by chunking in the batch
    dimension. The module is replicated on each machine and each device, and
    each such replica handles a portion of the input. During the backwards
    pass, gradients from each node are averaged.

    The batch size should be larger than the number of GPUs used locally.

    See also: :ref:`distributed-basics` and :ref:`cuda-nn-dataparallel-instead`.
    The same constraints on input as in :class:`torch.nn.DataParallel` apply.

    Creation of this class requires that ``torch.distributed`` to be already
    initialized, by calling :func:`torch.distributed.init_process_group`.

    ``DistributedDataParallel`` can be used in the following two ways:

    (1) Single-Process Multi-GPU

    In this case, a single process will be
    spawned on each host/node and each process will operate on all the GPUs
    of the node where it's running. To use ``DistributedDataParallel`` in
    this way, you can simply construct the model as the following:

        >>> torch.distributed.init_process_group(backend=\"nccl\")
        >>> model = DistributedDataParallel(model) # device_ids will include all GPU devices by default

    (2) Multi-Process Single-GPU

    This is the highly recommended way to use ``DistributedDataParallel``, with
    multiple processes, each of which operates on a single GPU. This is
    currently the fastest approach to do data parallel training using PyTorch
    and applies to both single-node(multi-GPU) and multi-node data
    parallel training. It is proven to be significantly faster than
    :class:`torch.nn.DataParallel` for single-node multi-GPU data
    parallel training.

    Here is how to use it: on each host with N GPUs, you should spawn up N
    processes, while ensuring that each process individually works on a single GPU
    from 0 to N-1. Therefore, it is your job to ensure that your training script
    operates on a single given GPU by calling:

        >>> torch.cuda.set_device(i)

    where i is from 0 to N-1. In each process, you should refer the following
    to construct this module:

        >>> torch.distributed.init_process_group(backend='nccl', world_size=4, init_method='...')
        >>> model = DistributedDataParallel(model, device_ids=[i], output_device=i)

    In order to spawn up multiple processes per node, you can use either
    ``torch.distributed.launch`` or ``torch.multiprocessing.spawn``

    .. note:: ``nccl`` backend is currently the fastest and
        highly recommended backend to be used with Multi-Process Single-GPU
        distributed training and this applies to both single-node and multi-node
        distributed training

    .. note:: This module also supports mixed-precision distributed training.
        This means that your model can have different types of parameters such
        as mixed types of fp16 and fp32, the gradient reduction on these
        mixed types of parameters will just work fine.
        Also note that ``nccl`` backend is currently the fastest and highly
        recommended backend for fp16/fp32 mixed-precision training.

    .. note:: If you use ``torch.save`` on one process to checkpoint the module,
        and ``torch.load`` on some other processes to recover it, make sure that
        ``map_location`` is configured properly for every process. Without
        ``map_location``, ``torch.load`` would recover the module to devices
        where the module was saved from.

    .. warning::
        This module works only with the ``gloo`` and ``nccl`` backends.

    .. warning::
        Constructor, forward method, and differentiation of the output (or a
        function of the output of this module) is a distributed synchronization
        point. Take that into account in case different processes might be
        executing different code.

    .. warning::
        This module assumes all parameters are registered in the model by the
        time it is created. No parameters should be added nor removed later.
        Same applies to buffers.

    .. warning::
        This module assumes all parameters are registered in the model of each
        distributed processes are in the same order. The module itself will
        conduct gradient all-reduction following the reverse order of the
        registered parameters of the model. In other words, it is users'
        responsibility to ensure that each distributed process has the exact
        same model and thus the exact same parameter registration order.

    .. warning::
        This module assumes all buffers and gradients are dense.

    .. warning::
        This module doesn't work with :func:`torch.autograd.grad` (i.e. it will
        only work if gradients are to be accumulated in ``.grad`` attributes of
        parameters).

    .. warning::

        If you plan on using this module with a ``nccl`` backend or a ``gloo``
        backend (that uses Infiniband), together with a DataLoader that uses
        multiple workers, please change the multiprocessing start method to
        ``forkserver`` (Python 3 only) or ``spawn``. Unfortunately
        Gloo (that uses Infiniband) and NCCL2 are not fork safe, and you will
        likely experience deadlocks if you don't change this setting.

    .. warning::
        Forward and backward hooks defined on :attr:`module` and its submodules
        won't be invoked anymore, unless the hooks are initialized in the
        :meth:`forward` method.

    .. warning::
        You should never try to change your model's parameters after wrapping
        up your model with DistributedDataParallel. In other words, when
        wrapping up your model with DistributedDataParallel, the constructor of
        DistributedDataParallel will register the additional gradient
        reduction functions on all the parameters of the model itself at the
        time of construction. If you change the model's parameters after
        the DistributedDataParallel construction, this is not supported and
        unexpected behaviors can happen, since some parameters' gradient
        reduction functions might not get called.

    .. note::
        Parameters are never broadcast between processes. The module performs
        an all-reduce step on gradients and assumes that they will be modified
        by the optimizer in all processes in the same way. Buffers
        (e.g. BatchNorm stats) are broadcast from the module in process of rank
        0, to all other replicas in the system in every iteration.

    Args:
        module (Module): module to be parallelized
        device_ids (list of int or torch.device): CUDA devices. This should
                   only be provided when the input module resides on a single
                   CUDA device. For single-device modules, the ``i``th
                   :attr:`module` replica is placed on ``device_ids[i]``. For
                   multi-device modules and CPU modules, device_ids must be None
                   or an empty list, and input data for the forward pass must be
                   placed on the correct device. (default: all devices for
                   single-device modules)
        output_device (int or torch.device): device location of output for
                      single-device CUDA modules. For multi-device modules and
                      CPU modules, it must be None, and the module itself
                      dictates the output location. (default: device_ids[0] for
                      single-device modules)
        broadcast_buffers (bool): flag that enables syncing (broadcasting) buffers of
                          the module at beginning of the forward function.
                          (default: ``True``)
        process_group: the process group to be used for distributed data
                       all-reduction. If ``None``, the default process group, which
                       is created by ```torch.distributed.init_process_group```,
                       will be used. (default: ``None``)
        bucket_cap_mb: DistributedDataParallel will bucket parameters into
                       multiple buckets so that gradient reduction of each
                       bucket can potentially overlap with backward computation.
                       :attr:`bucket_cap_mb` controls the bucket size in MegaBytes (MB)
                       (default: 25)
        find_unused_parameters (bool): Traverse the autograd graph of all tensors
                                       contained in the return value of the wrapped
                                       module's ``forward`` function.
                                       Parameters that don't receive gradients as
                                       part of this graph are preemptively marked
                                       as being ready to be reduced. Note that all
                                       ``forward`` outputs that are derived from
                                       module parameters must participate in
                                       calculating loss and later the gradient
                                       computation. If they don't, this wrapper will
                                       hang waiting for autograd to produce gradients
                                       for those parameters. Any outputs derived from
                                       module parameters that are otherwise unused can
                                       be detached from the autograd graph using
                                       ``torch.Tensor.detach``. (default: ``False``)
        check_reduction: when setting to ``True``, it enables DistributedDataParallel
                         to automatically check if the previous iteration's
                         backward reductions were successfully issued at the
                         beginning of every iteration's forward function.
                         You normally don't need this option enabled unless you
                         are observing weird behaviors such as different ranks
                         are getting different gradients, which should not
                         happen if DistributedDataParallel is correctly used.
                         (default: ``False``)

    Attributes:
        module (Module): the module to be parallelized

    Example::

        >>> torch.distributed.init_process_group(backend='nccl', world_size=4, init_method='...')
        >>> net = torch.nn.DistributedDataParallel(model, pg)
    "
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce parallel (import-module "torch.nn.parallel"))

(defn DistributedDataParallel 
  "Implements distributed data parallelism that is based on
    ``torch.distributed`` package at the module level.

    This container parallelizes the application of the given module by
    splitting the input across the specified devices by chunking in the batch
    dimension. The module is replicated on each machine and each device, and
    each such replica handles a portion of the input. During the backwards
    pass, gradients from each node are averaged.

    The batch size should be larger than the number of GPUs used locally.

    See also: :ref:`distributed-basics` and :ref:`cuda-nn-dataparallel-instead`.
    The same constraints on input as in :class:`torch.nn.DataParallel` apply.

    Creation of this class requires that ``torch.distributed`` to be already
    initialized, by calling :func:`torch.distributed.init_process_group`.

    ``DistributedDataParallel`` can be used in the following two ways:

    (1) Single-Process Multi-GPU

    In this case, a single process will be
    spawned on each host/node and each process will operate on all the GPUs
    of the node where it's running. To use ``DistributedDataParallel`` in
    this way, you can simply construct the model as the following:

        >>> torch.distributed.init_process_group(backend=\"nccl\")
        >>> model = DistributedDataParallel(model) # device_ids will include all GPU devices by default

    (2) Multi-Process Single-GPU

    This is the highly recommended way to use ``DistributedDataParallel``, with
    multiple processes, each of which operates on a single GPU. This is
    currently the fastest approach to do data parallel training using PyTorch
    and applies to both single-node(multi-GPU) and multi-node data
    parallel training. It is proven to be significantly faster than
    :class:`torch.nn.DataParallel` for single-node multi-GPU data
    parallel training.

    Here is how to use it: on each host with N GPUs, you should spawn up N
    processes, while ensuring that each process individually works on a single GPU
    from 0 to N-1. Therefore, it is your job to ensure that your training script
    operates on a single given GPU by calling:

        >>> torch.cuda.set_device(i)

    where i is from 0 to N-1. In each process, you should refer the following
    to construct this module:

        >>> torch.distributed.init_process_group(backend='nccl', world_size=4, init_method='...')
        >>> model = DistributedDataParallel(model, device_ids=[i], output_device=i)

    In order to spawn up multiple processes per node, you can use either
    ``torch.distributed.launch`` or ``torch.multiprocessing.spawn``

    .. note:: ``nccl`` backend is currently the fastest and
        highly recommended backend to be used with Multi-Process Single-GPU
        distributed training and this applies to both single-node and multi-node
        distributed training

    .. note:: This module also supports mixed-precision distributed training.
        This means that your model can have different types of parameters such
        as mixed types of fp16 and fp32, the gradient reduction on these
        mixed types of parameters will just work fine.
        Also note that ``nccl`` backend is currently the fastest and highly
        recommended backend for fp16/fp32 mixed-precision training.

    .. note:: If you use ``torch.save`` on one process to checkpoint the module,
        and ``torch.load`` on some other processes to recover it, make sure that
        ``map_location`` is configured properly for every process. Without
        ``map_location``, ``torch.load`` would recover the module to devices
        where the module was saved from.

    .. warning::
        This module works only with the ``gloo`` and ``nccl`` backends.

    .. warning::
        Constructor, forward method, and differentiation of the output (or a
        function of the output of this module) is a distributed synchronization
        point. Take that into account in case different processes might be
        executing different code.

    .. warning::
        This module assumes all parameters are registered in the model by the
        time it is created. No parameters should be added nor removed later.
        Same applies to buffers.

    .. warning::
        This module assumes all parameters are registered in the model of each
        distributed processes are in the same order. The module itself will
        conduct gradient all-reduction following the reverse order of the
        registered parameters of the model. In other words, it is users'
        responsibility to ensure that each distributed process has the exact
        same model and thus the exact same parameter registration order.

    .. warning::
        This module assumes all buffers and gradients are dense.

    .. warning::
        This module doesn't work with :func:`torch.autograd.grad` (i.e. it will
        only work if gradients are to be accumulated in ``.grad`` attributes of
        parameters).

    .. warning::

        If you plan on using this module with a ``nccl`` backend or a ``gloo``
        backend (that uses Infiniband), together with a DataLoader that uses
        multiple workers, please change the multiprocessing start method to
        ``forkserver`` (Python 3 only) or ``spawn``. Unfortunately
        Gloo (that uses Infiniband) and NCCL2 are not fork safe, and you will
        likely experience deadlocks if you don't change this setting.

    .. warning::
        Forward and backward hooks defined on :attr:`module` and its submodules
        won't be invoked anymore, unless the hooks are initialized in the
        :meth:`forward` method.

    .. warning::
        You should never try to change your model's parameters after wrapping
        up your model with DistributedDataParallel. In other words, when
        wrapping up your model with DistributedDataParallel, the constructor of
        DistributedDataParallel will register the additional gradient
        reduction functions on all the parameters of the model itself at the
        time of construction. If you change the model's parameters after
        the DistributedDataParallel construction, this is not supported and
        unexpected behaviors can happen, since some parameters' gradient
        reduction functions might not get called.

    .. note::
        Parameters are never broadcast between processes. The module performs
        an all-reduce step on gradients and assumes that they will be modified
        by the optimizer in all processes in the same way. Buffers
        (e.g. BatchNorm stats) are broadcast from the module in process of rank
        0, to all other replicas in the system in every iteration.

    Args:
        module (Module): module to be parallelized
        device_ids (list of int or torch.device): CUDA devices. This should
                   only be provided when the input module resides on a single
                   CUDA device. For single-device modules, the ``i``th
                   :attr:`module` replica is placed on ``device_ids[i]``. For
                   multi-device modules and CPU modules, device_ids must be None
                   or an empty list, and input data for the forward pass must be
                   placed on the correct device. (default: all devices for
                   single-device modules)
        output_device (int or torch.device): device location of output for
                      single-device CUDA modules. For multi-device modules and
                      CPU modules, it must be None, and the module itself
                      dictates the output location. (default: device_ids[0] for
                      single-device modules)
        broadcast_buffers (bool): flag that enables syncing (broadcasting) buffers of
                          the module at beginning of the forward function.
                          (default: ``True``)
        process_group: the process group to be used for distributed data
                       all-reduction. If ``None``, the default process group, which
                       is created by ```torch.distributed.init_process_group```,
                       will be used. (default: ``None``)
        bucket_cap_mb: DistributedDataParallel will bucket parameters into
                       multiple buckets so that gradient reduction of each
                       bucket can potentially overlap with backward computation.
                       :attr:`bucket_cap_mb` controls the bucket size in MegaBytes (MB)
                       (default: 25)
        find_unused_parameters (bool): Traverse the autograd graph of all tensors
                                       contained in the return value of the wrapped
                                       module's ``forward`` function.
                                       Parameters that don't receive gradients as
                                       part of this graph are preemptively marked
                                       as being ready to be reduced. Note that all
                                       ``forward`` outputs that are derived from
                                       module parameters must participate in
                                       calculating loss and later the gradient
                                       computation. If they don't, this wrapper will
                                       hang waiting for autograd to produce gradients
                                       for those parameters. Any outputs derived from
                                       module parameters that are otherwise unused can
                                       be detached from the autograd graph using
                                       ``torch.Tensor.detach``. (default: ``False``)
        check_reduction: when setting to ``True``, it enables DistributedDataParallel
                         to automatically check if the previous iteration's
                         backward reductions were successfully issued at the
                         beginning of every iteration's forward function.
                         You normally don't need this option enabled unless you
                         are observing weird behaviors such as different ranks
                         are getting different gradients, which should not
                         happen if DistributedDataParallel is correctly used.
                         (default: ``False``)

    Attributes:
        module (Module): the module to be parallelized

    Example::

        >>> torch.distributed.init_process_group(backend='nccl', world_size=4, init_method='...')
        >>> net = torch.nn.DistributedDataParallel(model, pg)
    "
  [module device_ids output_device & {:keys [dim broadcast_buffers process_group bucket_cap_mb find_unused_parameters check_reduction]
                       :or {dim 0 broadcast_buffers true bucket_cap_mb 25 find_unused_parameters false check_reduction false}} ]
    (py/call-attr-kw parallel "DistributedDataParallel" [module device_ids output_device] {:dim dim :broadcast_buffers broadcast_buffers :process_group process_group :bucket_cap_mb bucket_cap_mb :find_unused_parameters find_unused_parameters :check_reduction check_reduction }))

(defn add-module 
  "Adds a child module to the current module.

        The module can be accessed as an attribute using the given name.

        Args:
            name (string): name of the child module. The child module can be
                accessed from this module using the given name
            module (Module): child module to be added to the module.
        "
  [ self name module ]
  (py/call-attr self "add_module"  self name module ))

(defn apply 
  "Applies ``fn`` recursively to every submodule (as returned by ``.children()``)
        as well as self. Typical use includes initializing the parameters of a model
        (see also :ref:`torch-nn-init`).

        Args:
            fn (:class:`Module` -> None): function to be applied to each submodule

        Returns:
            Module: self

        Example::

            >>> def init_weights(m):
            >>>     print(m)
            >>>     if type(m) == nn.Linear:
            >>>         m.weight.data.fill_(1.0)
            >>>         print(m.weight)
            >>> net = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2))
            >>> net.apply(init_weights)
            Linear(in_features=2, out_features=2, bias=True)
            Parameter containing:
            tensor([[ 1.,  1.],
                    [ 1.,  1.]])
            Linear(in_features=2, out_features=2, bias=True)
            Parameter containing:
            tensor([[ 1.,  1.],
                    [ 1.,  1.]])
            Sequential(
              (0): Linear(in_features=2, out_features=2, bias=True)
              (1): Linear(in_features=2, out_features=2, bias=True)
            )
            Sequential(
              (0): Linear(in_features=2, out_features=2, bias=True)
              (1): Linear(in_features=2, out_features=2, bias=True)
            )
        "
  [ self fn ]
  (py/call-attr self "apply"  self fn ))

(defn buffers 
  "Returns an iterator over module buffers.

        Args:
            recurse (bool): if True, then yields buffers of this module
                and all submodules. Otherwise, yields only buffers that
                are direct members of this module.

        Yields:
            torch.Tensor: module buffer

        Example::

            >>> for buf in model.buffers():
            >>>     print(type(buf.data), buf.size())
            <class 'torch.FloatTensor'> (20L,)
            <class 'torch.FloatTensor'> (20L, 1L, 5L, 5L)

        "
  [self  & {:keys [recurse]
                       :or {recurse true}} ]
    (py/call-attr-kw self "buffers" [] {:recurse recurse }))

(defn children 
  "Returns an iterator over immediate children modules.

        Yields:
            Module: a child module
        "
  [ self  ]
  (py/call-attr self "children"  self  ))

(defn cpu 
  "Moves all model parameters and buffers to the CPU.

        Returns:
            Module: self
        "
  [ self  ]
  (py/call-attr self "cpu"  self  ))

(defn cuda 
  "Moves all model parameters and buffers to the GPU.

        This also makes associated parameters and buffers different objects. So
        it should be called before constructing optimizer if the module will
        live on GPU while being optimized.

        Arguments:
            device (int, optional): if specified, all parameters will be
                copied to that device

        Returns:
            Module: self
        "
  [ self device ]
  (py/call-attr self "cuda"  self device ))

(defn double 
  "Casts all floating point parameters and buffers to ``double`` datatype.

        Returns:
            Module: self
        "
  [ self  ]
  (py/call-attr self "double"  self  ))

(defn eval 
  "Sets the module in evaluation mode.

        This has any effect only on certain modules. See documentations of
        particular modules for details of their behaviors in training/evaluation
        mode, if they are affected, e.g. :class:`Dropout`, :class:`BatchNorm`,
        etc.

        This is equivalent with :meth:`self.train(False) <torch.nn.Module.train>`.

        Returns:
            Module: self
        "
  [ self  ]
  (py/call-attr self "eval"  self  ))

(defn extra-repr 
  "Set the extra representation of the module

        To print customized extra information, you should reimplement
        this method in your own modules. Both single-line and multi-line
        strings are acceptable.
        "
  [ self  ]
  (py/call-attr self "extra_repr"  self  ))

(defn float 
  "Casts all floating point parameters and buffers to float datatype.

        Returns:
            Module: self
        "
  [ self  ]
  (py/call-attr self "float"  self  ))

(defn forward 
  ""
  [ self  ]
  (py/call-attr self "forward"  self  ))

(defn gather 
  ""
  [ self outputs output_device ]
  (py/call-attr self "gather"  self outputs output_device ))

(defn half 
  "Casts all floating point parameters and buffers to ``half`` datatype.

        Returns:
            Module: self
        "
  [ self  ]
  (py/call-attr self "half"  self  ))

(defn load-state-dict 
  "Copies parameters and buffers from :attr:`state_dict` into
        this module and its descendants. If :attr:`strict` is ``True``, then
        the keys of :attr:`state_dict` must exactly match the keys returned
        by this module's :meth:`~torch.nn.Module.state_dict` function.

        Arguments:
            state_dict (dict): a dict containing parameters and
                persistent buffers.
            strict (bool, optional): whether to strictly enforce that the keys
                in :attr:`state_dict` match the keys returned by this module's
                :meth:`~torch.nn.Module.state_dict` function. Default: ``True``

        Returns:
            ``NamedTuple`` with ``missing_keys`` and ``unexpected_keys`` fields:
                * **missing_keys** is a list of str containing the missing keys
                * **unexpected_keys** is a list of str containing the unexpected keys
        "
  [self state_dict & {:keys [strict]
                       :or {strict true}} ]
    (py/call-attr-kw self "load_state_dict" [state_dict] {:strict strict }))

(defn modules 
  "Returns an iterator over all modules in the network.

        Yields:
            Module: a module in the network

        Note:
            Duplicate modules are returned only once. In the following
            example, ``l`` will be returned only once.

        Example::

            >>> l = nn.Linear(2, 2)
            >>> net = nn.Sequential(l, l)
            >>> for idx, m in enumerate(net.modules()):
                    print(idx, '->', m)

            0 -> Sequential(
              (0): Linear(in_features=2, out_features=2, bias=True)
              (1): Linear(in_features=2, out_features=2, bias=True)
            )
            1 -> Linear(in_features=2, out_features=2, bias=True)

        "
  [ self  ]
  (py/call-attr self "modules"  self  ))

(defn named-buffers 
  "Returns an iterator over module buffers, yielding both the
        name of the buffer as well as the buffer itself.

        Args:
            prefix (str): prefix to prepend to all buffer names.
            recurse (bool): if True, then yields buffers of this module
                and all submodules. Otherwise, yields only buffers that
                are direct members of this module.

        Yields:
            (string, torch.Tensor): Tuple containing the name and buffer

        Example::

            >>> for name, buf in self.named_buffers():
            >>>    if name in ['running_var']:
            >>>        print(buf.size())

        "
  [self  & {:keys [prefix recurse]
                       :or {prefix "" recurse true}} ]
    (py/call-attr-kw self "named_buffers" [] {:prefix prefix :recurse recurse }))

(defn named-children 
  "Returns an iterator over immediate children modules, yielding both
        the name of the module as well as the module itself.

        Yields:
            (string, Module): Tuple containing a name and child module

        Example::

            >>> for name, module in model.named_children():
            >>>     if name in ['conv4', 'conv5']:
            >>>         print(module)

        "
  [ self  ]
  (py/call-attr self "named_children"  self  ))

(defn named-modules 
  "Returns an iterator over all modules in the network, yielding
        both the name of the module as well as the module itself.

        Yields:
            (string, Module): Tuple of name and module

        Note:
            Duplicate modules are returned only once. In the following
            example, ``l`` will be returned only once.

        Example::

            >>> l = nn.Linear(2, 2)
            >>> net = nn.Sequential(l, l)
            >>> for idx, m in enumerate(net.named_modules()):
                    print(idx, '->', m)

            0 -> ('', Sequential(
              (0): Linear(in_features=2, out_features=2, bias=True)
              (1): Linear(in_features=2, out_features=2, bias=True)
            ))
            1 -> ('0', Linear(in_features=2, out_features=2, bias=True))

        "
  [self memo & {:keys [prefix]
                       :or {prefix ""}} ]
    (py/call-attr-kw self "named_modules" [memo] {:prefix prefix }))

(defn named-parameters 
  "Returns an iterator over module parameters, yielding both the
        name of the parameter as well as the parameter itself.

        Args:
            prefix (str): prefix to prepend to all parameter names.
            recurse (bool): if True, then yields parameters of this module
                and all submodules. Otherwise, yields only parameters that
                are direct members of this module.

        Yields:
            (string, Parameter): Tuple containing the name and parameter

        Example::

            >>> for name, param in self.named_parameters():
            >>>    if name in ['bias']:
            >>>        print(param.size())

        "
  [self  & {:keys [prefix recurse]
                       :or {prefix "" recurse true}} ]
    (py/call-attr-kw self "named_parameters" [] {:prefix prefix :recurse recurse }))

(defn no-sync 
  "
        A context manager to disable gradient synchronizations across DDP
        processes. Within this context, gradients will be accumulated on module
        variables, which will later be synchronized in the first
        forward-backward pass exiting the context.

        Example::

            >>> ddp = torch.nn.DistributedDataParallel(model, pg)
            >>> with ddp.no_sync():
            ...   for input in inputs:
            ...     ddp(input).backward()  # no synchronization, accumulate grads
            ... ddp(another_input).backward()  # synchronize grads
        "
  [ self  ]
  (py/call-attr self "no_sync"  self  ))

(defn parallel-apply 
  ""
  [ self replicas inputs kwargs ]
  (py/call-attr self "parallel_apply"  self replicas inputs kwargs ))

(defn parameters 
  "Returns an iterator over module parameters.

        This is typically passed to an optimizer.

        Args:
            recurse (bool): if True, then yields parameters of this module
                and all submodules. Otherwise, yields only parameters that
                are direct members of this module.

        Yields:
            Parameter: module parameter

        Example::

            >>> for param in model.parameters():
            >>>     print(type(param.data), param.size())
            <class 'torch.FloatTensor'> (20L,)
            <class 'torch.FloatTensor'> (20L, 1L, 5L, 5L)

        "
  [self  & {:keys [recurse]
                       :or {recurse true}} ]
    (py/call-attr-kw self "parameters" [] {:recurse recurse }))

(defn register-backward-hook 
  "Registers a backward hook on the module.

        The hook will be called every time the gradients with respect to module
        inputs are computed. The hook should have the following signature::

            hook(module, grad_input, grad_output) -> Tensor or None

        The :attr:`grad_input` and :attr:`grad_output` may be tuples if the
        module has multiple inputs or outputs. The hook should not modify its
        arguments, but it can optionally return a new gradient with respect to
        input that will be used in place of :attr:`grad_input` in subsequent
        computations.

        Returns:
            :class:`torch.utils.hooks.RemovableHandle`:
                a handle that can be used to remove the added hook by calling
                ``handle.remove()``

        .. warning ::

            The current implementation will not have the presented behavior
            for complex :class:`Module` that perform many operations.
            In some failure cases, :attr:`grad_input` and :attr:`grad_output` will only
            contain the gradients for a subset of the inputs and outputs.
            For such :class:`Module`, you should use :func:`torch.Tensor.register_hook`
            directly on a specific input or output to get the required gradients.

        "
  [ self hook ]
  (py/call-attr self "register_backward_hook"  self hook ))

(defn register-buffer 
  "Adds a persistent buffer to the module.

        This is typically used to register a buffer that should not to be
        considered a model parameter. For example, BatchNorm's ``running_mean``
        is not a parameter, but is part of the persistent state.

        Buffers can be accessed as attributes using given names.

        Args:
            name (string): name of the buffer. The buffer can be accessed
                from this module using the given name
            tensor (Tensor): buffer to be registered.

        Example::

            >>> self.register_buffer('running_mean', torch.zeros(num_features))

        "
  [ self name tensor ]
  (py/call-attr self "register_buffer"  self name tensor ))

(defn register-forward-hook 
  "Registers a forward hook on the module.

        The hook will be called every time after :func:`forward` has computed an output.
        It should have the following signature::

            hook(module, input, output) -> None or modified output

        The hook can modify the output. It can modify the input inplace but
        it will not have effect on forward since this is called after
        :func:`forward` is called.

        Returns:
            :class:`torch.utils.hooks.RemovableHandle`:
                a handle that can be used to remove the added hook by calling
                ``handle.remove()``
        "
  [ self hook ]
  (py/call-attr self "register_forward_hook"  self hook ))

(defn register-forward-pre-hook 
  "Registers a forward pre-hook on the module.

        The hook will be called every time before :func:`forward` is invoked.
        It should have the following signature::

            hook(module, input) -> None or modified input

        The hook can modify the input. User can either return a tuple or a
        single modified value in the hook. We will wrap the value into a tuple
        if a single value is returned(unless that value is already a tuple).

        Returns:
            :class:`torch.utils.hooks.RemovableHandle`:
                a handle that can be used to remove the added hook by calling
                ``handle.remove()``
        "
  [ self hook ]
  (py/call-attr self "register_forward_pre_hook"  self hook ))

(defn register-parameter 
  "Adds a parameter to the module.

        The parameter can be accessed as an attribute using given name.

        Args:
            name (string): name of the parameter. The parameter can be accessed
                from this module using the given name
            param (Parameter): parameter to be added to the module.
        "
  [ self name param ]
  (py/call-attr self "register_parameter"  self name param ))

(defn requires-grad- 
  "Change if autograd should record operations on parameters in this
        module.

        This method sets the parameters' :attr:`requires_grad` attributes
        in-place.

        This method is helpful for freezing part of the module for finetuning
        or training parts of a model individually (e.g., GAN training).

        Args:
            requires_grad (bool): whether autograd should record operations on
                                  parameters in this module. Default: ``True``.

        Returns:
            Module: self
        "
  [self  & {:keys [requires_grad]
                       :or {requires_grad true}} ]
    (py/call-attr-kw self "requires_grad_" [] {:requires_grad requires_grad }))

(defn scatter 
  ""
  [ self inputs kwargs device_ids ]
  (py/call-attr self "scatter"  self inputs kwargs device_ids ))

(defn share-memory 
  ""
  [ self  ]
  (py/call-attr self "share_memory"  self  ))

(defn state-dict 
  "Returns a dictionary containing a whole state of the module.

        Both parameters and persistent buffers (e.g. running averages) are
        included. Keys are corresponding parameter and buffer names.

        Returns:
            dict:
                a dictionary containing a whole state of the module

        Example::

            >>> module.state_dict().keys()
            ['bias', 'weight']

        "
  [self destination & {:keys [prefix keep_vars]
                       :or {prefix "" keep_vars false}} ]
    (py/call-attr-kw self "state_dict" [destination] {:prefix prefix :keep_vars keep_vars }))

(defn to 
  "Moves and/or casts the parameters and buffers.

        This can be called as

        .. function:: to(device=None, dtype=None, non_blocking=False)

        .. function:: to(dtype, non_blocking=False)

        .. function:: to(tensor, non_blocking=False)

        Its signature is similar to :meth:`torch.Tensor.to`, but only accepts
        floating point desired :attr:`dtype` s. In addition, this method will
        only cast the floating point parameters and buffers to :attr:`dtype`
        (if given). The integral parameters and buffers will be moved
        :attr:`device`, if that is given, but with dtypes unchanged. When
        :attr:`non_blocking` is set, it tries to convert/move asynchronously
        with respect to the host if possible, e.g., moving CPU Tensors with
        pinned memory to CUDA devices.

        See below for examples.

        .. note::
            This method modifies the module in-place.

        Args:
            device (:class:`torch.device`): the desired device of the parameters
                and buffers in this module
            dtype (:class:`torch.dtype`): the desired floating point type of
                the floating point parameters and buffers in this module
            tensor (torch.Tensor): Tensor whose dtype and device are the desired
                dtype and device for all parameters and buffers in this module

        Returns:
            Module: self

        Example::

            >>> linear = nn.Linear(2, 2)
            >>> linear.weight
            Parameter containing:
            tensor([[ 0.1913, -0.3420],
                    [-0.5113, -0.2325]])
            >>> linear.to(torch.double)
            Linear(in_features=2, out_features=2, bias=True)
            >>> linear.weight
            Parameter containing:
            tensor([[ 0.1913, -0.3420],
                    [-0.5113, -0.2325]], dtype=torch.float64)
            >>> gpu1 = torch.device(\"cuda:1\")
            >>> linear.to(gpu1, dtype=torch.half, non_blocking=True)
            Linear(in_features=2, out_features=2, bias=True)
            >>> linear.weight
            Parameter containing:
            tensor([[ 0.1914, -0.3420],
                    [-0.5112, -0.2324]], dtype=torch.float16, device='cuda:1')
            >>> cpu = torch.device(\"cpu\")
            >>> linear.to(cpu)
            Linear(in_features=2, out_features=2, bias=True)
            >>> linear.weight
            Parameter containing:
            tensor([[ 0.1914, -0.3420],
                    [-0.5112, -0.2324]], dtype=torch.float16)

        "
  [ self  ]
  (py/call-attr self "to"  self  ))

(defn train 
  ""
  [self  & {:keys [mode]
                       :or {mode true}} ]
    (py/call-attr-kw self "train" [] {:mode mode }))

(defn type 
  "Casts all parameters and buffers to :attr:`dst_type`.

        Arguments:
            dst_type (type or string): the desired type

        Returns:
            Module: self
        "
  [ self dst_type ]
  (py/call-attr self "type"  self dst_type ))

(defn zero-grad 
  "Sets gradients of all model parameters to zero."
  [ self  ]
  (py/call-attr self "zero_grad"  self  ))
