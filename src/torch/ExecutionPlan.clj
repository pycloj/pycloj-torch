(ns torch.ExecutionPlan
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
