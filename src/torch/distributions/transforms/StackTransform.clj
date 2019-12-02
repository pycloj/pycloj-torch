(ns torch.distributions.transforms.StackTransform
  "
    Transform functor that applies a sequence of transforms `tseq`
    component-wise to each submatrix at `dim`
    in a way compatible with :func:`torch.stack`.

    Example::
       x = torch.stack([torch.range(1, 10), torch.range(1, 10)], dim=1)
       t = StackTransform([ExpTransform(), identity_transform], dim=1)
       y = t(x)
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

(defn StackTransform 
  "
    Transform functor that applies a sequence of transforms `tseq`
    component-wise to each submatrix at `dim`
    in a way compatible with :func:`torch.stack`.

    Example::
       x = torch.stack([torch.range(1, 10), torch.range(1, 10)], dim=1)
       t = StackTransform([ExpTransform(), identity_transform], dim=1)
       y = t(x)
    "
  [tseq & {:keys [dim]
                       :or {dim 0}} ]
    (py/call-attr-kw transforms "StackTransform" [tseq] {:dim dim }))

(defn bijective 
  ""
  [ self ]
    (py/call-attr self "bijective"))

(defn codomain 
  ""
  [ self ]
    (py/call-attr self "codomain"))

(defn domain 
  ""
  [ self ]
    (py/call-attr self "domain"))

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

(defn sign 
  "
        Returns the sign of the determinant of the Jacobian, if applicable.
        In general this only makes sense for bijective transforms.
        "
  [ self ]
    (py/call-attr self "sign"))
