(ns torch.cuda.sparse.ByteTensor
  ""
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce sparse (import-module "torch.cuda.sparse"))

(defn align-to 
  "Permutes the dimensions of the :attr:`self` tensor to match the order
        specified in :attr:`names`, adding size-one dims for any new names.

        All of the dims of :attr:`self` must be named in order to use this method.
        The resulting tensor is a view on the original tensor.

        All dimension names of :attr:`self` must be present in :attr:`names`.
        :attr:`names` may contain additional names that are not in ``self.names``;
        the output tensor has a size-one dimension for each of those new names.

        :attr:`names` may contain up to one Ellipsis (``...``).
        The Ellipsis is expanded to be equal to all dimension names of :attr:`self`
        that are not mentioned in :attr:`names`, in the order that they appear
        in :attr:`self`.

        Python 2 does not support Ellipsis but one may use a string literal
        instead (``'...'``).

        Arguments:
            names (iterable of str): The desired dimension ordering of the
                output tensor. May contain up to one Ellipsis that is expanded
                to all unmentioned dim names of :attr:`self`.

        Examples::

            >>> tensor = torch.randn(2, 2, 2, 2, 2, 2)
            >>> named_tensor = tensor.refine_names('A', 'B', 'C', 'D', 'E', 'F')

            # Move the F and E dims to the front while keeping the rest in order
            >>> named_tensor.align_to('F', 'E', ...)

        .. warning::
            The named tensor API is experimental and subject to change.

        "
  [ self  ]
  (py/call-attr self "align_to"  self  ))

(defn backward 
  "Computes the gradient of current tensor w.r.t. graph leaves.

        The graph is differentiated using the chain rule. If the tensor is
        non-scalar (i.e. its data has more than one element) and requires
        gradient, the function additionally requires specifying ``gradient``.
        It should be a tensor of matching type and location, that contains
        the gradient of the differentiated function w.r.t. ``self``.

        This function accumulates gradients in the leaves - you might need to
        zero them before calling it.

        Arguments:
            gradient (Tensor or None): Gradient w.r.t. the
                tensor. If it is a tensor, it will be automatically converted
                to a Tensor that does not require grad unless ``create_graph`` is True.
                None values can be specified for scalar Tensors or ones that
                don't require grad. If a None value would be acceptable then
                this argument is optional.
            retain_graph (bool, optional): If ``False``, the graph used to compute
                the grads will be freed. Note that in nearly all cases setting
                this option to True is not needed and often can be worked around
                in a much more efficient way. Defaults to the value of
                ``create_graph``.
            create_graph (bool, optional): If ``True``, graph of the derivative will
                be constructed, allowing to compute higher order derivative
                products. Defaults to ``False``.
        "
  [self gradient retain_graph & {:keys [create_graph]
                       :or {create_graph false}} ]
    (py/call-attr-kw self "backward" [gradient retain_graph] {:create_graph create_graph }))

(defn is-shared 
  "Checks if tensor is in shared memory.

        This is always ``True`` for CUDA tensors.
        "
  [ self  ]
  (py/call-attr self "is_shared"  self  ))

(defn lu 
  "See :func:`torch.lu`"
  [self  & {:keys [pivot get_infos]
                       :or {pivot true get_infos false}} ]
    (py/call-attr-kw self "lu" [] {:pivot pivot :get_infos get_infos }))

(defn norm 
  "See :func:`torch.norm`"
  [self  & {:keys [p dim keepdim dtype]
                       :or {p "fro" keepdim false}} ]
    (py/call-attr-kw self "norm" [] {:p p :dim dim :keepdim keepdim :dtype dtype }))

(defn refine-names 
  "Refines the dimension names of :attr:`self` according to :attr:`names`.

        Refining is a special case of renaming that \"lifts\" unnamed dimensions.
        A ``None`` dim can be refined to have any name; a named dim can only be
        refined to have the same name.

        Because named tensors can coexist with unnamed tensors, refining names
        gives a nice way to write named-tensor-aware code that works with both
        named and unnamed tensors.

        :attr:`names` may contain up to one Ellipsis (``...``).
        The Ellipsis is expanded greedily; it is expanded in-place to fill
        :attr:`names` to the same length as ``self.dim()`` using names from the
        corresponding indices of ``self.names``.

        Python 2 does not support Ellipsis but one may use a string literal
        instead (``'...'``).

        Arguments:
            names (iterable of str): The desired names of the output tensor. May
                contain up to one Ellipsis.

        Examples::

            >>> imgs = torch.randn(32, 3, 128, 128)
            >>> named_imgs = imgs.refine_names('N', 'C', 'H', 'W')
            >>> named_imgs.names
            ('N', 'C', 'H', 'W')

            >>> tensor = torch.randn(2, 3, 5, 7, 11)
            >>> tensor = tensor.refine_names('A', ..., 'B', 'C')
            >>> tensor.names
            ('A', None, None, 'B', 'C')

        .. warning::
            The named tensor API is experimental and subject to change.

        "
  [ self  ]
  (py/call-attr self "refine_names"  self  ))

(defn register-hook 
  "Registers a backward hook.

        The hook will be called every time a gradient with respect to the
        Tensor is computed. The hook should have the following signature::

            hook(grad) -> Tensor or None


        The hook should not modify its argument, but it can optionally return
        a new gradient which will be used in place of :attr:`grad`.

        This function returns a handle with a method ``handle.remove()``
        that removes the hook from the module.

        Example::

            >>> v = torch.tensor([0., 0., 0.], requires_grad=True)
            >>> h = v.register_hook(lambda grad: grad * 2)  # double the gradient
            >>> v.backward(torch.tensor([1., 2., 3.]))
            >>> v.grad

             2
             4
             6
            [torch.FloatTensor of size (3,)]

            >>> h.remove()  # removes the hook
        "
  [ self hook ]
  (py/call-attr self "register_hook"  self hook ))

(defn reinforce 
  ""
  [ self reward ]
  (py/call-attr self "reinforce"  self reward ))

(defn rename 
  "Renames dimension names of :attr:`self`.

        There are two main usages:

        ``self.rename(**rename_map)`` returns a view on tensor that has dims
        renamed as specified in the mapping :attr:`rename_map`.

        ``self.rename(*names)`` returns a view on tensor, renaming all
        dimensions positionally using :attr:`names`.
        Use ``self.rename(None)`` to drop names on a tensor.

        One cannot specify both positional args :attr:`names` and keyword args
        :attr:`rename_map`.

        Examples::

            >>> imgs = torch.rand(2, 3, 5, 7, names=('N', 'C', 'H', 'W'))
            >>> renamed_imgs = imgs.rename(N='batch', C='channels')
            >>> renamed_imgs.names
            ('batch', 'channels', 'H', 'W')

            >>> renamed_imgs = imgs.rename(None)
            >>> renamed_imgs.names
            (None,)

            >>> renamed_imgs = imgs.rename('batch', 'channel', 'height', 'width')
            >>> renamed_imgs.names
            ('batch', 'channel', 'height', 'width')

        .. warning::
            The named tensor API is experimental and subject to change.

        "
  [ self  ]
  (py/call-attr self "rename"  self  ))

(defn rename- 
  "In-place version of :meth:`~Tensor.rename`."
  [ self  ]
  (py/call-attr self "rename_"  self  ))

(defn resize 
  ""
  [ self  ]
  (py/call-attr self "resize"  self  ))

(defn resize-as 
  ""
  [ self tensor ]
  (py/call-attr self "resize_as"  self tensor ))

(defn retain-grad 
  "Enables .grad attribute for non-leaf Tensors."
  [ self  ]
  (py/call-attr self "retain_grad"  self  ))

(defn share-memory- 
  "Moves the underlying storage to shared memory.

        This is a no-op if the underlying storage is already in shared memory
        and for CUDA tensors. Tensors in shared memory cannot be resized.
        "
  [ self  ]
  (py/call-attr self "share_memory_"  self  ))

(defn split 
  "See :func:`torch.split`
        "
  [self split_size & {:keys [dim]
                       :or {dim 0}} ]
    (py/call-attr-kw self "split" [split_size] {:dim dim }))

(defn stft 
  "See :func:`torch.stft`

        .. warning::
          This function changed signature at version 0.4.1. Calling with
          the previous signature may cause error or return incorrect result.
        "
  [self n_fft hop_length win_length window & {:keys [center pad_mode normalized onesided]
                       :or {center true pad_mode "reflect" normalized false onesided true}} ]
    (py/call-attr-kw self "stft" [n_fft hop_length win_length window] {:center center :pad_mode pad_mode :normalized normalized :onesided onesided }))

(defn unflatten 
  "Unflattens the named dimension :attr:`dim`, viewing it in the shape
        specified by :attr:`namedshape`.

        Arguments:
            namedshape: (iterable of ``(name, size)`` tuples).

        Examples::

            >>> flat_imgs = torch.rand(32, 3 * 128 * 128, names=('N', 'features'))
            >>> imgs = flat_imgs.unflatten('features', (('C', 3), ('H', 128), ('W', 128)))
            >>> imgs.names, images.shape
            (('N', 'C', 'H', 'W'), torch.Size([32, 3, 128, 128]))

        .. warning::
            The named tensor API is experimental and subject to change.

        "
  [ self dim namedshape ]
  (py/call-attr self "unflatten"  self dim namedshape ))

(defn unique 
  "Returns the unique elements of the input tensor.

        See :func:`torch.unique`
        "
  [self  & {:keys [sorted return_inverse return_counts dim]
                       :or {sorted true return_inverse false return_counts false}} ]
    (py/call-attr-kw self "unique" [] {:sorted sorted :return_inverse return_inverse :return_counts return_counts :dim dim }))

(defn unique-consecutive 
  "Eliminates all but the first element from every consecutive group of equivalent elements.

        See :func:`torch.unique_consecutive`
        "
  [self  & {:keys [return_inverse return_counts dim]
                       :or {return_inverse false return_counts false}} ]
    (py/call-attr-kw self "unique_consecutive" [] {:return_inverse return_inverse :return_counts return_counts :dim dim }))
