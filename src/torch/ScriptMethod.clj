(ns torch.ScriptMethod
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

(defn code 
  ""
  [ self ]
    (py/call-attr self "code"))

(defn graph 
  ""
  [ self ]
    (py/call-attr self "graph"))

(defn graph-for 
  ""
  [ self  ]
  (py/call-attr self "graph_for"  self  ))

(defn name 
  ""
  [ self ]
    (py/call-attr self "name"))

(defn schema 
  ""
  [ self ]
    (py/call-attr self "schema"))
