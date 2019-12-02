(ns torch.distributions.transforms.PowerTransform
  "
    Transform via the mapping :math:`y = x^{\text{exponent}}`.
    "
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce transforms (import-module "torch.distributions.transforms"))

(defn PowerTransform 
  "
    Transform via the mapping :math:`y = x^{\text{exponent}}`.
    "
  [exponent & {:keys [cache_size]
                       :or {cache_size 0}} ]
    (py/call-attr-kw transforms "PowerTransform" [exponent] {:cache_size cache_size }))

(defn inv 
  "
        Returns the inverse :class:`Transform` of this transform.
        This should satisfy ``t.inv.inv is t``.
        "
  [ self ]
    (py/call-attr self "inv"))

(defn log-abs-det-jacobian 
  ""
  [ self x y ]
  (py/call-attr self "log_abs_det_jacobian"  self x y ))
