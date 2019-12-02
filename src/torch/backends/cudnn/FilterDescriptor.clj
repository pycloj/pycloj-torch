(ns torch.backends.cudnn.FilterDescriptor
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

(defn FilterDescriptor 
  ""
  [  ]
  (py/call-attr m "FilterDescriptor"  ))

(defn as-tuple 
  ""
  [ self  ]
  (py/call-attr self "as_tuple"  self  ))

(defn set 
  ""
  [ self weight ]
  (py/call-attr self "set"  self weight ))
