(ns torch.nn.utils.clip-grad
  ""
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce clip-grad (import-module "torch.nn.utils.clip_grad"))

(defn clip-grad-norm 
  "Clips gradient norm of an iterable of parameters.

    .. warning::
        This method is now deprecated in favor of
        :func:`torch.nn.utils.clip_grad_norm_`.
    "
  [parameters max_norm & {:keys [norm_type]
                       :or {norm_type 2}} ]
    (py/call-attr-kw clip-grad "clip_grad_norm" [parameters max_norm] {:norm_type norm_type }))

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
    (py/call-attr-kw clip-grad "clip_grad_norm_" [parameters max_norm] {:norm_type norm_type }))

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
  (py/call-attr clip-grad "clip_grad_value_"  parameters clip_value ))
