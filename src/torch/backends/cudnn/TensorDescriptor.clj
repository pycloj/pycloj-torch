(ns torch.backends.cudnn.TensorDescriptor
  ""
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce cudnn (import-module "torch.backends.cudnn"))

(defn TensorDescriptor 
  ""
  [  ]
  (py/call-attr m "TensorDescriptor"  ))

(defn as-tuple 
  ""
  [ self  ]
  (py/call-attr self "as_tuple"  self  ))

(defn set 
  ""
  [ self tensor ]
  (py/call-attr self "set"  self tensor ))
