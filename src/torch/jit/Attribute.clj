(ns torch.jit.Attribute
  "Attribute(value, type)"
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce jit (import-module "torch.jit"))

(defn Attribute 
  "Attribute(value, type)"
  [ value type ]
  (py/call-attr jit "Attribute"  value type ))

(defn type 
  "Alias for field number 1"
  [ self ]
    (py/call-attr self "type"))

(defn value 
  "Alias for field number 0"
  [ self ]
    (py/call-attr self "value"))
