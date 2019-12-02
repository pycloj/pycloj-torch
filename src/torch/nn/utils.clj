(ns torch.nn.utils
  ""
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce utils (import-module "torch.nn.utils"))

(defn clip-grad-norm 
  "Clips gradient norm of an iterable of parameters.

    .. warning::
        This method is now deprecated in favor of
        :func:`torch.nn.utils.clip_grad_norm_`.
    "
  [parameters max_norm & {:keys [norm_type]
                       :or {norm_type 2}} ]
    (py/call-attr-kw utils "clip_grad_norm" [parameters max_norm] {:norm_type norm_type }))

(defn clip-grad-norm- 
  "Clips gradient norm of an iterable of parameters.

    The norm is computed over all gradients together, as if they were
    concatenated into a single vector. Gradients are modified in-place.

    Arguments:
        parameters (Iterable[Tensor] or Tensor): an iterable of Tensors or a
            single Tensor that will have gradients normalized
        max_norm (float or int): max norm of the gradients
        norm_type (float or int): type of the used p-norm. Can be ``'inf'`` for
            infinity norm.

    Returns:
        Total norm of the parameters (viewed as a single vector).
    "
  [parameters max_norm & {:keys [norm_type]
                       :or {norm_type 2}} ]
    (py/call-attr-kw utils "clip_grad_norm_" [parameters max_norm] {:norm_type norm_type }))

(defn clip-grad-value- 
  "Clips gradient of an iterable of parameters at specified value.

    Gradients are modified in-place.

    Arguments:
        parameters (Iterable[Tensor] or Tensor): an iterable of Tensors or a
            single Tensor that will have gradients normalized
        clip_value (float or int): maximum allowed value of the gradients.
            The gradients are clipped in the range
            :math:`\left[\text{-clip\_value}, \text{clip\_value}\right]`
    "
  [ parameters clip_value ]
  (py/call-attr utils "clip_grad_value_"  parameters clip_value ))

(defn fuse-conv-bn-eval 
  ""
  [ conv bn ]
  (py/call-attr utils "fuse_conv_bn_eval"  conv bn ))

(defn fuse-conv-bn-weights 
  ""
  [ conv_w conv_b bn_rm bn_rv bn_eps bn_w bn_b ]
  (py/call-attr utils "fuse_conv_bn_weights"  conv_w conv_b bn_rm bn_rv bn_eps bn_w bn_b ))

(defn parameters-to-vector 
  "Convert parameters to one vector

    Arguments:
        parameters (Iterable[Tensor]): an iterator of Tensors that are the
            parameters of a model.

    Returns:
        The parameters represented by a single vector
    "
  [ parameters ]
  (py/call-attr utils "parameters_to_vector"  parameters ))

(defn remove-spectral-norm 
  "Removes the spectral normalization reparameterization from a module.

    Args:
        module (Module): containing module
        name (str, optional): name of weight parameter

    Example:
        >>> m = spectral_norm(nn.Linear(40, 10))
        >>> remove_spectral_norm(m)
    "
  [module & {:keys [name]
                       :or {name "weight"}} ]
    (py/call-attr-kw utils "remove_spectral_norm" [module] {:name name }))

(defn remove-weight-norm 
  "Removes the weight normalization reparameterization from a module.

    Args:
        module (Module): containing module
        name (str, optional): name of weight parameter

    Example:
        >>> m = weight_norm(nn.Linear(20, 40))
        >>> remove_weight_norm(m)
    "
  [module & {:keys [name]
                       :or {name "weight"}} ]
    (py/call-attr-kw utils "remove_weight_norm" [module] {:name name }))

(defn spectral-norm 
  "Applies spectral normalization to a parameter in the given module.

    .. math::
        \mathbf{W}_{SN} = \dfrac{\mathbf{W}}{\sigma(\mathbf{W})},
        \sigma(\mathbf{W}) = \max_{\mathbf{h}: \mathbf{h} \ne 0} \dfrac{\|\mathbf{W} \mathbf{h}\|_2}{\|\mathbf{h}\|_2}

    Spectral normalization stabilizes the training of discriminators (critics)
    in Generative Adversarial Networks (GANs) by rescaling the weight tensor
    with spectral norm :math:`\sigma` of the weight matrix calculated using
    power iteration method. If the dimension of the weight tensor is greater
    than 2, it is reshaped to 2D in power iteration method to get spectral
    norm. This is implemented via a hook that calculates spectral norm and
    rescales weight before every :meth:`~Module.forward` call.

    See `Spectral Normalization for Generative Adversarial Networks`_ .

    .. _`Spectral Normalization for Generative Adversarial Networks`: https://arxiv.org/abs/1802.05957

    Args:
        module (nn.Module): containing module
        name (str, optional): name of weight parameter
        n_power_iterations (int, optional): number of power iterations to
            calculate spectral norm
        eps (float, optional): epsilon for numerical stability in
            calculating norms
        dim (int, optional): dimension corresponding to number of outputs,
            the default is ``0``, except for modules that are instances of
            ConvTranspose{1,2,3}d, when it is ``1``

    Returns:
        The original module with the spectral norm hook

    Example::

        >>> m = spectral_norm(nn.Linear(20, 40))
        >>> m
        Linear(in_features=20, out_features=40, bias=True)
        >>> m.weight_u.size()
        torch.Size([40])

    "
  [module & {:keys [name n_power_iterations eps dim]
                       :or {name "weight" n_power_iterations 1 eps 1e-12}} ]
    (py/call-attr-kw utils "spectral_norm" [module] {:name name :n_power_iterations n_power_iterations :eps eps :dim dim }))

(defn vector-to-parameters 
  "Convert one vector to the parameters

    Arguments:
        vec (Tensor): a single vector represents the parameters of a model.
        parameters (Iterable[Tensor]): an iterator of Tensors that are the
            parameters of a model.
    "
  [ vec parameters ]
  (py/call-attr utils "vector_to_parameters"  vec parameters ))

(defn weight-norm 
  "Applies weight normalization to a parameter in the given module.

    .. math::
         \mathbf{w} = g \dfrac{\mathbf{v}}{\|\mathbf{v}\|}

    Weight normalization is a reparameterization that decouples the magnitude
    of a weight tensor from its direction. This replaces the parameter specified
    by :attr:`name` (e.g. ``'weight'``) with two parameters: one specifying the magnitude
    (e.g. ``'weight_g'``) and one specifying the direction (e.g. ``'weight_v'``).
    Weight normalization is implemented via a hook that recomputes the weight
    tensor from the magnitude and direction before every :meth:`~Module.forward`
    call.

    By default, with ``dim=0``, the norm is computed independently per output
    channel/plane. To compute a norm over the entire weight tensor, use
    ``dim=None``.

    See https://arxiv.org/abs/1602.07868

    Args:
        module (Module): containing module
        name (str, optional): name of weight parameter
        dim (int, optional): dimension over which to compute the norm

    Returns:
        The original module with the weight norm hook

    Example::

        >>> m = weight_norm(nn.Linear(20, 40), name='weight')
        >>> m
        Linear(in_features=20, out_features=40, bias=True)
        >>> m.weight_g.size()
        torch.Size([40, 1])
        >>> m.weight_v.size()
        torch.Size([40, 20])

    "
  [module & {:keys [name dim]
                       :or {name "weight" dim 0}} ]
    (py/call-attr-kw utils "weight_norm" [module] {:name name :dim dim }))
