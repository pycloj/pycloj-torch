(ns torch.FunctionSchema
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

(defn arguments 
  ""
  [ self ]
    (py/call-attr self "arguments"))

(defn name 
  ""
  [ self ]
    (py/call-attr self "name"))

(defn overload-name 
  ""
  [ self ]
    (py/call-attr self "overload_name"))

(defn returns 
  ""
  [ self ]
    (py/call-attr self "returns"))
