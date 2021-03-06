(ns torch.distributions.transforms.AbsTransform
  "
    Transform via the mapping :math:`y = |x|`.
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

(defn AbsTransform 
  "
    Transform via the mapping :math:`y = |x|`.
    "
  [ & {:keys [cache_size]
       :or {cache_size 0}} ]
  
   (py/call-attr-kw transforms "AbsTransform" [] {:cache_size cache_size }))

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
