(ns torch.GraphExecutorState
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

(defn execution-plans 
  ""
  [ self ]
    (py/call-attr self "execution_plans"))

(defn fallback 
  ""
  [ self ]
    (py/call-attr self "fallback"))

(defn graph 
  ""
  [ self ]
    (py/call-attr self "graph"))
