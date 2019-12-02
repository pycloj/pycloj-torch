(ns torch.jit.Function
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

(defn qualified-name 
  ""
  [ self ]
    (py/call-attr self "qualified_name"))

(defn schema 
  ""
  [ self ]
    (py/call-attr self "schema"))
