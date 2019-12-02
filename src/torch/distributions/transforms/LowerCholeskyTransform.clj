(ns torch.distributions.transforms.LowerCholeskyTransform
  "
    Transform from unconstrained matrices to lower-triangular matrices with
    nonnegative diagonal entries.

    This is useful for parameterizing positive definite matrices in terms of
    their Cholesky factorization.
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

(defn LowerCholeskyTransform 
  "
    Transform from unconstrained matrices to lower-triangular matrices with
    nonnegative diagonal entries.

    This is useful for parameterizing positive definite matrices in terms of
    their Cholesky factorization.
    "
  [ & {:keys [cache_size]
       :or {cache_size 0}} ]
  
   (py/call-attr-kw transforms "LowerCholeskyTransform" [] {:cache_size cache_size }))

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
