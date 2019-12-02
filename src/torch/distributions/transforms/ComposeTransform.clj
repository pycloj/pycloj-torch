(ns torch.distributions.transforms.ComposeTransform
  "
    Composes multiple transforms in a chain.
    The transforms being composed are responsible for caching.

    Args:
        parts (list of :class:`Transform`): A list of transforms to compose.
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

(defn ComposeTransform 
  "
    Composes multiple transforms in a chain.
    The transforms being composed are responsible for caching.

    Args:
        parts (list of :class:`Transform`): A list of transforms to compose.
    "
  [ parts ]
  (py/call-attr transforms "ComposeTransform"  parts ))

(defn codomain 
  ""
  [ self ]
    (py/call-attr self "codomain"))

(defn domain 
  ""
  [ self ]
    (py/call-attr self "domain"))

(defn inv 
  ""
  [ self ]
    (py/call-attr self "inv"))

(defn log-abs-det-jacobian 
  ""
  [ self x y ]
  (py/call-attr self "log_abs_det_jacobian"  self x y ))
