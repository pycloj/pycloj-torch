(ns torch.Argument
  ""
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce torch (import-module "torch"))

(defn N 
  ""
  [ self ]
    (py/call-attr self "N"))

(defn default-value 
  ""
  [ self ]
    (py/call-attr self "default_value"))

(defn name 
  ""
  [ self ]
    (py/call-attr self "name"))

(defn type 
  ""
  [ self ]
    (py/call-attr self "type"))
