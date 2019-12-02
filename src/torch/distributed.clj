(ns torch.distributed
  ""
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce distributed (import-module "torch.distributed"))

(defn all-gather 
  "
    Gathers tensors from the whole group in a list.

    Arguments:
        tensor_list (list[Tensor]): Output list. It should contain
            correctly-sized tensors to be used for output of the collective.
        tensor (Tensor): Tensor to be broadcast from current process.
        group (ProcessGroup, optional): The process group to work on
        async_op (bool, optional): Whether this op should be an async op

    Returns:
        Async work handle, if async_op is set to True.
        None, if not async_op or if not part of the group

    "
  [tensor_list tensor & {:keys [group async_op]
                       :or {group <object object at 0x121e82100> async_op false}} ]
    (py/call-attr-kw distributed "all_gather" [tensor_list tensor] {:group group :async_op async_op }))

(defn all-gather-multigpu 
  "
    Gathers tensors from the whole group in a list.
    Each tensor in ``tensor_list`` should reside on a separate GPU

    Only nccl backend is currently supported
    tensors should only be GPU tensors

    Arguments:
        output_tensor_lists (List[List[Tensor]]): Output lists. It should
            contain correctly-sized tensors on each GPU to be used for output
            of the collective, e.g. ``output_tensor_lists[i]`` contains the
            all_gather result that resides on the GPU of
            ``input_tensor_list[i]``.

            Note that each element of ``output_tensor_lists`` has the size of
            ``world_size * len(input_tensor_list)``, since the function all
            gathers the result from every single GPU in the group. To interpret
            each element of ``output_tensor_lists[i]``, note that
            ``input_tensor_list[j]`` of rank k will be appear in
            ``output_tensor_lists[i][k * world_size + j]``

            Also note that ``len(output_tensor_lists)``, and the size of each
            element in ``output_tensor_lists`` (each element is a list,
            therefore ``len(output_tensor_lists[i])``) need to be the same
            for all the distributed processes calling this function.

        input_tensor_list (List[Tensor]): List of tensors(on different GPUs) to
            be broadcast from current process.
            Note that ``len(input_tensor_list)`` needs to be the same for
            all the distributed processes calling this function.

        group (ProcessGroup, optional): The process group to work on
        async_op (bool, optional): Whether this op should be an async op

    Returns:
        Async work handle, if async_op is set to True.
        None, if not async_op or if not part of the group

    "
  [output_tensor_lists input_tensor_list & {:keys [group async_op]
                       :or {group <object object at 0x121e82100> async_op false}} ]
    (py/call-attr-kw distributed "all_gather_multigpu" [output_tensor_lists input_tensor_list] {:group group :async_op async_op }))

(defn all-reduce 
  "
    Reduces the tensor data across all machines in such a way that all get
    the final result.

    After the call ``tensor`` is going to be bitwise identical in all processes.

    Arguments:
        tensor (Tensor): Input and output of the collective. The function
            operates in-place.
        op (optional): One of the values from
            ``torch.distributed.ReduceOp``
            enum.  Specifies an operation used for element-wise reductions.
        group (ProcessGroup, optional): The process group to work on
        async_op (bool, optional): Whether this op should be an async op

    Returns:
        Async work handle, if async_op is set to True.
        None, if not async_op or if not part of the group

    "
  [tensor & {:keys [op group async_op]
                       :or {op ReduceOp.SUM group <object object at 0x121e82100> async_op false}} ]
    (py/call-attr-kw distributed "all_reduce" [tensor] {:op op :group group :async_op async_op }))

(defn all-reduce-coalesced 
  "
    WARNING: at this time individual shape checking is not implemented across nodes.
    For example, if the rank 0 node passes [torch.rand(4), torch.rand(2)] and the
    rank 1 node passes [torch.rand(2), torch.rand(2), torch.rand(2)], the allreduce
    operation will proceed without complaint and return erroneous outputs. This lack
    of shape checking results in significant performance improvements but users of this
    function should take extra care to ensure that each node passes in tensors whose
    shapes match across nodes.

    Reduces each tensor in tensors (residing on the same device) across all machines
    in such a way that all get the final result.

    After the call each tensor in tensors is going to bitwise identical
    in all processes.

    Arguments:
        tensors (List[Tensor]): Input and output of the collective. The function
            operates in-place.
        op (Optional[ReduceOp]): One of the values from
            ``torch.distributed.ReduceOp`` enum. Specifies an operation used for
            element-wise reductions.
        group (Optional[ProcessGroup]): The process group to work on.
        async_op (Optional[bool]): Whether this op should be an async op.

    Returns:
        Async work handle, if async_op is set to True.
        None, if not async_op or if not part of the group.

    "
  [tensors & {:keys [op group async_op]
                       :or {op ReduceOp.SUM group <object object at 0x121e82100> async_op false}} ]
    (py/call-attr-kw distributed "all_reduce_coalesced" [tensors] {:op op :group group :async_op async_op }))

(defn all-reduce-multigpu 
  "
    Reduces the tensor data across all machines in such a way that all get
    the final result. This function reduces a number of tensors on every node,
    while each tensor resides on different GPUs.
    Therefore, the input tensor in the tensor list needs to be GPU tensors.
    Also, each tensor in the tensor list needs to reside on a different GPU.

    After the call, all ``tensor`` in ``tensor_list`` is going to be bitwise
    identical in all processes.

    Only nccl and gloo backend is currently supported
    tensors should only be GPU tensors

    Arguments:
        tensor list (List[Tensor]): List of input and output tensors of
            the collective. The function operates in-place and requires that
            each tensor to be a GPU tensor on different GPUs.
            You also need to make sure that ``len(tensor_list)`` is the same for
            all the distributed processes calling this function.
        op (optional): One of the values from
            ``torch.distributed.ReduceOp``
            enum.  Specifies an operation used for element-wise reductions.
        group (ProcessGroup, optional): The process group to work on
        async_op (bool, optional): Whether this op should be an async op

    Returns:
        Async work handle, if async_op is set to True.
        None, if not async_op or if not part of the group

    "
  [tensor_list & {:keys [op group async_op]
                       :or {op ReduceOp.SUM group <object object at 0x121e82100> async_op false}} ]
    (py/call-attr-kw distributed "all_reduce_multigpu" [tensor_list] {:op op :group group :async_op async_op }))

(defn barrier 
  "
    Synchronizes all processes.

    This collective blocks processes until the whole group enters this function,
    if async_op is False, or if async work handle is called on wait().

    Arguments:
        group (ProcessGroup, optional): The process group to work on
        async_op (bool, optional): Whether this op should be an async op

    Returns:
        Async work handle, if async_op is set to True.
        None, if not async_op or if not part of the group
    "
  [ & {:keys [group async_op]
       :or {group <object object at 0x121e82100> async_op false}} ]
  
   (py/call-attr-kw distributed "barrier" [] {:group group :async_op async_op }))

(defn broadcast 
  "
    Broadcasts the tensor to the whole group.

    ``tensor`` must have the same number of elements in all processes
    participating in the collective.

    Arguments:
        tensor (Tensor): Data to be sent if ``src`` is the rank of current
            process, and tensor to be used to save received data otherwise.
        src (int): Source rank.
        group (ProcessGroup, optional): The process group to work on
        async_op (bool, optional): Whether this op should be an async op

    Returns:
        Async work handle, if async_op is set to True.
        None, if not async_op or if not part of the group

    "
  [tensor src & {:keys [group async_op]
                       :or {group <object object at 0x121e82100> async_op false}} ]
    (py/call-attr-kw distributed "broadcast" [tensor src] {:group group :async_op async_op }))

(defn broadcast-multigpu 
  "
    Broadcasts the tensor to the whole group with multiple GPU tensors
    per node.

    ``tensor`` must have the same number of elements in all the GPUs from
    all processes participating in the collective. each tensor in the list must
    be on a different GPU

    Only nccl and gloo backend are currently supported
    tensors should only be GPU tensors

    Arguments:
        tensor_list (List[Tensor]): Tensors that participate in the collective
            operation. If ``src`` is the rank, then the specified ``src_tensor``
            element of ``tensor_list`` (``tensor_list[src_tensor]``) will be
            broadcast to all other tensors (on different GPUs) in the src process
            and all tensors in ``tensor_list`` of other non-src processes.
            You also need to make sure that ``len(tensor_list)`` is the same
            for all the distributed processes calling this function.

        src (int): Source rank.
        group (ProcessGroup, optional): The process group to work on
        async_op (bool, optional): Whether this op should be an async op
        src_tensor (int, optional): Source tensor rank within ``tensor_list``

    Returns:
        Async work handle, if async_op is set to True.
        None, if not async_op or if not part of the group

    "
  [tensor_list src & {:keys [group async_op src_tensor]
                       :or {group <object object at 0x121e82100> async_op false src_tensor 0}} ]
    (py/call-attr-kw distributed "broadcast_multigpu" [tensor_list src] {:group group :async_op async_op :src_tensor src_tensor }))

(defn destroy-process-group 
  "
    Destroy a given process group, and deinitialize the distributed package

    Arguments:
        group (ProcessGroup, optional): The process group to be destroyed, if
                                        group.WORLD is given, all process
                                        groups including the default one will
                                        be destroyed.
    "
  [ & {:keys [group]
       :or {group <object object at 0x121e82100>}} ]
  
   (py/call-attr-kw distributed "destroy_process_group" [] {:group group }))

(defn gather 
  "
    Gathers a list of tensors in a single process.

    Arguments:
        tensor (Tensor): Input tensor.
        gather_list (list[Tensor], optional): List of appropriately-sized
            tensors to use for gathered data (default is None, must be specified
            on the destination rank)
        dst (int, optional): Destination rank (default is 0)
        group (ProcessGroup, optional): The process group to work on
        async_op (bool, optional): Whether this op should be an async op

    Returns:
        Async work handle, if async_op is set to True.
        None, if not async_op or if not part of the group

    "
  [tensor gather_list & {:keys [dst group async_op]
                       :or {dst 0 group <object object at 0x121e82100> async_op false}} ]
    (py/call-attr-kw distributed "gather" [tensor gather_list] {:dst dst :group group :async_op async_op }))

(defn get-backend 
  "
    Returns the backend of the given process group.

    Arguments:
        group (ProcessGroup, optional): The process group to work on. The
            default is the general main process group. If another specific group
            is specified, the calling process must be part of :attr:`group`.

    Returns:
        The backend of the given process group as a lower case string.

    "
  [ & {:keys [group]
       :or {group <object object at 0x121e82100>}} ]
  
   (py/call-attr-kw distributed "get_backend" [] {:group group }))

(defn get-rank 
  "
    Returns the rank of current process group

    Rank is a unique identifier assigned to each process within a distributed
    process group. They are always consecutive integers ranging from 0 to
    ``world_size``.

    Arguments:
        group (ProcessGroup, optional): The process group to work on

    Returns:
        The rank of the process group
        -1, if not part of the group

    "
  [ & {:keys [group]
       :or {group <object object at 0x121e82100>}} ]
  
   (py/call-attr-kw distributed "get_rank" [] {:group group }))

(defn get-worker-id 
  ""
  [  ]
  (py/call-attr distributed "get_worker_id"  ))

(defn get-world-size 
  "
    Returns the number of processes in the current process group

    Arguments:
        group (ProcessGroup, optional): The process group to work on

    Returns:
        The world size of the process group
        -1, if not part of the group

    "
  [ & {:keys [group]
       :or {group <object object at 0x121e82100>}} ]
  
   (py/call-attr-kw distributed "get_world_size" [] {:group group }))

(defn init-model-parallel 
  "
            Initializes model parallel primitives such as the local rpc agent
            and distributed autograd.

            Initializes the local RPC agent which immediately makes the current
            process ready to send and receive RPCs. The caller needs to make
            sure the specified backend is properly intialized before calling
            this method. For example, to use ``pg`` (ProcessGroup) backend,
            ``init_process_group`` must be invoked prior to this method.

            Arguments:
                backend (Enum): type of RPC backend implementation.
                            Currently, process group backend is the only
                            available backend implementation. (default:
                            ``RpcBackend.PROCESS_GROUP``).
                self_name (str): a globally unique name of this node. (e.g.,
                            ``Trainer3``, ``ParameterServer2``, ``Master``,
                            ``Worker1``) Name can only contain number, alphabet,
                            underscore, and/or dash, and must be shorter than
                            128 characters.
                self_rank (int): a globally unique id/rank of this node.
                init_method(str): backend specific init arguments.
                num_send_recv_threads(int): Number of threads for send/recv work.
            "
  [self_name & {:keys [backend self_rank init_method num_send_recv_threads]
                       :or {backend RpcBackend.PROCESS_GROUP self_rank -1 num_send_recv_threads 4}} ]
    (py/call-attr-kw distributed "init_model_parallel" [self_name] {:backend backend :self_rank self_rank :init_method init_method :num_send_recv_threads num_send_recv_threads }))

(defn init-process-group 
  "
    Initializes the default distributed process group, and this will also
    initialize the distributed package.

    There are 2 main ways to initialize a process group:
        1. Specify ``store``, ``rank``, and ``world_size`` explicitly.
        2. Specify ``init_method`` (a URL string) which indicates where/how
           to discover peers. Optionally specify ``rank`` and ``world_size``,
           or encode all required parameters in the URL and omit them.
        If neither is specified, ``init_method`` is assumed to be \"env://\".


    Arguments:
        backend (str or Backend): The backend to use. Depending on
            build-time configurations, valid values include ``mpi``, ``gloo``,
            and ``nccl``. This field should be given as a lowercase string
            (e.g., ``\"gloo\"``), which can also be accessed via
            :class:`Backend` attributes (e.g., ``Backend.GLOO``). If using
            multiple processes per machine with ``nccl`` backend, each process
            must have exclusive access to every GPU it uses, as sharing GPUs
            between processes can result in deadlocks.
        init_method (str, optional): URL specifying how to initialize the
                                     process group. Default is \"env://\" if no
                                     ``init_method`` or ``store`` is specified.
                                     Mutually exclusive with ``store``.
        world_size (int, optional): Number of processes participating in
                                    the job. Required if ``store`` is specified.
        rank (int, optional): Rank of the current process.
                              Required if ``store`` is specified.
        store(Store, optional): Key/value store accessible to all workers, used
                                to exchange connection/address information.
                                Mutually exclusive with ``init_method``.
        timeout (timedelta, optional): Timeout for operations executed against
            the process group. Default value equals 30 minutes.
            This is only applicable for the ``gloo`` backend.
        group_name (str, optional, deprecated): Group name.

    To enable ``backend == Backend.MPI``, PyTorch needs to built from source
    on a system that supports MPI. The same applies to NCCL as well.

    "
  [backend init_method & {:keys [timeout world_size rank store group_name]
                       :or {timeout 0:30:00 world_size -1 rank -1 group_name ""}} ]
    (py/call-attr-kw distributed "init_process_group" [backend init_method] {:timeout timeout :world_size world_size :rank rank :store store :group_name group_name }))

(defn init-rpc-backend 
  ""
  [ backend_name ]
  (py/call-attr distributed "init_rpc_backend"  backend_name ))

(defn irecv 
  "
    Receives a tensor asynchronously.

    Arguments:
        tensor (Tensor): Tensor to fill with received data.
        src (int): Source rank.
        group (ProcessGroup, optional): The process group to work on
        tag (int, optional): Tag to match recv with remote send

    Returns:
        A distributed request object.
        None, if not part of the group

    "
  [tensor src & {:keys [group tag]
                       :or {group <object object at 0x121e82100> tag 0}} ]
    (py/call-attr-kw distributed "irecv" [tensor src] {:group group :tag tag }))

(defn is-available 
  ""
  [  ]
  (py/call-attr distributed "is_available"  ))

(defn is-gloo-available 
  "
    Checks if the Gloo backend is available.

    "
  [  ]
  (py/call-attr distributed "is_gloo_available"  ))

(defn is-initialized 
  "
    Checking if the default process group has been initialized

    "
  [  ]
  (py/call-attr distributed "is_initialized"  ))

(defn is-mpi-available 
  "
    Checks if the MPI backend is available.

    "
  [  ]
  (py/call-attr distributed "is_mpi_available"  ))

(defn is-nccl-available 
  "
    Checks if the NCCL backend is available.

    "
  [  ]
  (py/call-attr distributed "is_nccl_available"  ))

(defn is-rpc-backend-registered 
  ""
  [ backend_name ]
  (py/call-attr distributed "is_rpc_backend_registered"  backend_name ))

(defn isend 
  "
    Sends a tensor asynchronously.

    Arguments:
        tensor (Tensor): Tensor to send.
        dst (int): Destination rank.
        group (ProcessGroup, optional): The process group to work on
        tag (int, optional): Tag to match send with remote recv

    Returns:
        A distributed request object.
        None, if not part of the group

    "
  [tensor dst & {:keys [group tag]
                       :or {group <object object at 0x121e82100> tag 0}} ]
    (py/call-attr-kw distributed "isend" [tensor dst] {:group group :tag tag }))

(defn join-rpc 
  "
    Block until all local and remote RPC processes reach this method, process
    (send and receive) all pending messages, and then destroy local RPC agent.
    Every RPC process must call this method before exit.
    "
  [  ]
  (py/call-attr distributed "join_rpc"  ))

(defn new-group 
  "
    Creates a new distributed group.

    This function requires that all processes in the main group (i.e. all
    processes that are part of the distributed job) enter this function, even
    if they are not going to be members of the group. Additionally, groups
    should be created in the same order in all processes.

    Arguments:
        ranks (list[int]): List of ranks of group members.
        timeout (timedelta, optional): Timeout for operations executed against
            the process group. Default value equals 30 minutes.
            This is only applicable for the ``gloo`` backend.
        backend (str or Backend, optional): The backend to use. Depending on
            build-time configurations, valid values are ``gloo`` and ``nccl``.
            By default uses the same backend as the global group. This field
            should be given as a lowercase string (e.g., ``\"gloo\"``), which can
            also be accessed via :class:`Backend` attributes (e.g.,
            ``Backend.GLOO``).

    Returns:
        A handle of distributed group that can be given to collective calls.
    "
  [ranks & {:keys [timeout backend]
                       :or {timeout 0:30:00}} ]
    (py/call-attr-kw distributed "new_group" [ranks] {:timeout timeout :backend backend }))

(defn recv 
  "
    Receives a tensor synchronously.

    Arguments:
        tensor (Tensor): Tensor to fill with received data.
        src (int, optional): Source rank. Will receive from any
            process if unspecified.
        group (ProcessGroup, optional): The process group to work on
        tag (int, optional): Tag to match recv with remote send

    Returns:
        Sender rank
        -1, if not part of the group

    "
  [tensor src & {:keys [group tag]
                       :or {group <object object at 0x121e82100> tag 0}} ]
    (py/call-attr-kw distributed "recv" [tensor src] {:group group :tag tag }))

(defn reduce 
  "
    Reduces the tensor data across all machines.

    Only the process with rank ``dst`` is going to receive the final result.

    Arguments:
        tensor (Tensor): Input and output of the collective. The function
            operates in-place.
        dst (int): Destination rank
        op (optional): One of the values from
            ``torch.distributed.ReduceOp``
            enum.  Specifies an operation used for element-wise reductions.
        group (ProcessGroup, optional): The process group to work on
        async_op (bool, optional): Whether this op should be an async op

    Returns:
        Async work handle, if async_op is set to True.
        None, if not async_op or if not part of the group

    "
  [tensor dst & {:keys [op group async_op]
                       :or {op ReduceOp.SUM group <object object at 0x121e82100> async_op false}} ]
    (py/call-attr-kw distributed "reduce" [tensor dst] {:op op :group group :async_op async_op }))

(defn reduce-multigpu 
  "
    Reduces the tensor data on multiple GPUs across all machines. Each tensor
    in ``tensor_list`` should reside on a separate GPU

    Only the GPU of ``tensor_list[dst_tensor]`` on the process with rank ``dst``
    is going to receive the final result.

    Only nccl backend is currently supported
    tensors should only be GPU tensors

    Arguments:
        tensor_list (List[Tensor]): Input and output GPU tensors of the
            collective. The function operates in-place.
            You also need to make sure that ``len(tensor_list)`` is the same for
            all the distributed processes calling this function.
        dst (int): Destination rank
        op (optional): One of the values from
            ``torch.distributed.ReduceOp``
            enum.  Specifies an operation used for element-wise reductions.
        group (ProcessGroup, optional): The process group to work on
        async_op (bool, optional): Whether this op should be an async op
        dst_tensor (int, optional): Destination tensor rank within
                                    ``tensor_list``

    Returns:
        Async work handle, if async_op is set to True.
        None, otherwise

    "
  [tensor_list dst & {:keys [op group async_op dst_tensor]
                       :or {op ReduceOp.SUM group <object object at 0x121e82100> async_op false dst_tensor 0}} ]
    (py/call-attr-kw distributed "reduce_multigpu" [tensor_list dst] {:op op :group group :async_op async_op :dst_tensor dst_tensor }))

(defn reduce-scatter 
  "
    Reduces, then scatters a list of tensors to all processes in a group.

    Arguments:
        output (Tensor): Output tensor.
        input_list (list[Tensor]): List of tensors to reduce and scatter.
        group (ProcessGroup, optional): The process group to work on.
        async_op (bool, optional): Whether this op should be an async op.

    Returns:
        Async work handle, if async_op is set to True.
        None, if not async_op or if not part of the group.

    "
  [output input_list & {:keys [op group async_op]
                       :or {op ReduceOp.SUM group <object object at 0x121e82100> async_op false}} ]
    (py/call-attr-kw distributed "reduce_scatter" [output input_list] {:op op :group group :async_op async_op }))

(defn reduce-scatter-multigpu 
  "
    Reduce and scatter a list of tensors to the whole group.  Only nccl backend
    is currently supported.

    Each tensor in ``output_tensor_list`` should reside on a separate GPU, as
    should each list of tensors in ``input_tensor_lists``.

    Arguments:
        output_tensor_list (List[Tensor]): Output tensors (on different GPUs)
            to receive the result of the operation.

            Note that ``len(output_tensor_list)`` needs to be the same for all
            the distributed processes calling this function.

        input_tensor_lists (List[List[Tensor]]): Input lists.  It should
            contain correctly-sized tensors on each GPU to be used for input of
            the collective, e.g. ``input_tensor_lists[i]`` contains the
            reduce_scatter input that resides on the GPU of
            ``output_tensor_list[i]``.

            Note that each element of ``input_tensor_lists`` has the size of
            ``world_size * len(output_tensor_list)``, since the function
            scatters the result from every single GPU in the group.  To
            interpret each element of ``input_tensor_lists[i]``, note that
            ``output_tensor_list[j]`` of rank k receives the reduce-scattered
            result from ``input_tensor_lists[i][k * world_size + j]``

            Also note that ``len(input_tensor_lists)``, and the size of each
            element in ``input_tensor_lists`` (each element is a list,
            therefore ``len(input_tensor_lists[i])``) need to be the same for
            all the distributed processes calling this function.

        group (ProcessGroup, optional): The process group to work on.
        async_op (bool, optional): Whether this op should be an async op.

    Returns:
        Async work handle, if async_op is set to True.
        None, if not async_op or if not part of the group.

    "
  [output_tensor_list input_tensor_lists & {:keys [op group async_op]
                       :or {op ReduceOp.SUM group <object object at 0x121e82100> async_op false}} ]
    (py/call-attr-kw distributed "reduce_scatter_multigpu" [output_tensor_list input_tensor_lists] {:op op :group group :async_op async_op }))

(defn register-rendezvous-handler 
  "Registers a new rendezvous handler.

    Before we can run collective algorithms, participating processes
    need to find each other and exchange information to be able to
    communicate. We call this process rendezvous.

    The outcome of the rendezvous process is a triplet containing a
    shared key/value store, the rank of the process, and the total
    number of participating processes.

    If none of the bundled rendezvous methods apply to your execution
    environment you can opt to register your own rendezvous handler.
    Pick a unique name and use the URL scheme to identify it when
    calling the `rendezvous()` function.

    Arguments:
        scheme (str): URL scheme to identify your rendezvous handler.
        handler (function): Handler that is invoked when the
            `rendezvous()` function is called with a URL that uses
            the corresponding scheme. It must be a generator function
            that yields the triplet.
    "
  [ scheme handler ]
  (py/call-attr distributed "register_rendezvous_handler"  scheme handler ))

(defn remote 
  ""
  [  ]
  (py/call-attr distributed "remote"  ))

(defn rendezvous 
  ""
  [ url ]
  (py/call-attr distributed "rendezvous"  url ))

(defn rpc 
  ""
  [  ]
  (py/call-attr distributed "rpc"  ))

(defn scatter 
  "
    Scatters a list of tensors to all processes in a group.

    Each process will receive exactly one tensor and store its data in the
    ``tensor`` argument.

    Arguments:
        tensor (Tensor): Output tensor.
        scatter_list (list[Tensor]): List of tensors to scatter (default is
            None, must be specified on the source rank)
        src (int): Source rank (default is 0)
        group (ProcessGroup, optional): The process group to work on
        async_op (bool, optional): Whether this op should be an async op

    Returns:
        Async work handle, if async_op is set to True.
        None, if not async_op or if not part of the group

    "
  [tensor scatter_list & {:keys [src group async_op]
                       :or {src 0 group <object object at 0x121e82100> async_op false}} ]
    (py/call-attr-kw distributed "scatter" [tensor scatter_list] {:src src :group group :async_op async_op }))

(defn send 
  "
    Sends a tensor synchronously.

    Arguments:
        tensor (Tensor): Tensor to send.
        dst (int): Destination rank.
        group (ProcessGroup, optional): The process group to work on
        tag (int, optional): Tag to match send with remote recv

    "
  [tensor dst & {:keys [group tag]
                       :or {group <object object at 0x121e82100> tag 0}} ]
    (py/call-attr-kw distributed "send" [tensor dst] {:group group :tag tag }))

(defn serialize 
  ""
  [ obj ]
  (py/call-attr distributed "serialize"  obj ))

(defn sync-rpc 
  ""
  [  ]
  (py/call-attr distributed "sync_rpc"  ))
