(ns torch.distributions.relaxed-categorical.ExpTransform
  "
    Transform via the mapping :math:`y = \exp(x)`.
    "
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce relaxed-categorical (import-module "torch.distributions.relaxed_categorical"))

(defn ExpTransform 
  "
    Transform via the mapping :math:`y = \exp(x)`.
    "
  [ & {:keys [cache_size]
       :or {cache_size 0}} ]
  
   (py/call-attr-kw relaxed-categorical "ExpTransform" [] {:cache_size cache_size }))

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
