(ns torch.distributions.SoftmaxTransform
  "
    Transform from unconstrained space to the simplex via :math:`y = \exp(x)` then
    normalizing.

    This is not bijective and cannot be used for HMC. However this acts mostly
    coordinate-wise (except for the final normalization), and thus is
    appropriate for coordinate-wise optimization algorithms.
    "
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce distributions (import-module "torch.distributions"))

(defn SoftmaxTransform 
  "
    Transform from unconstrained space to the simplex via :math:`y = \exp(x)` then
    normalizing.

    This is not bijective and cannot be used for HMC. However this acts mostly
    coordinate-wise (except for the final normalization), and thus is
    appropriate for coordinate-wise optimization algorithms.
    "
  [ & {:keys [cache_size]
       :or {cache_size 0}} ]
  
   (py/call-attr-kw distributions "SoftmaxTransform" [] {:cache_size cache_size }))

(defn inv 
  "
        Returns the inverse :class:`Transform` of this transform.
        This should satisfy ``t.inv.inv is t``.
        "
  [ self ]
    (py/call-attr self "inv"))

(defn log-abs-det-jacobian 
  "
        Computes the log det jacobian `log |dy/dx|` given input and output.
        "
  [ self x y ]
  (py/call-attr self "log_abs_det_jacobian"  self x y ))

(defn sign 
  "
        Returns the sign of the determinant of the Jacobian, if applicable.
        In general this only makes sense for bijective transforms.
        "
  [ self ]
    (py/call-attr self "sign"))
