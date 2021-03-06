(ns torch.jit.OrderedDictWrapper
  ""
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce jit (import-module "torch.jit"))

(defn OrderedDictWrapper 
  ""
  [ module ]
  (py/call-attr jit "OrderedDictWrapper"  module ))

(defn items 
  ""
  [ self  ]
  (py/call-attr self "items"  self  ))

(defn keys 
  ""
  [ self  ]
  (py/call-attr self "keys"  self  ))

(defn values 
  ""
  [ self  ]
  (py/call-attr self "values"  self  ))
