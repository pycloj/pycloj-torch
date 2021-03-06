(ns torch.distributed.AllreduceCoalescedOptions
  ""
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce distributed (import-module "torch.distributed"))

(defn reduceOp 
  ""
  [ self ]
    (py/call-attr self "reduceOp"))

(defn timeout 
  ""
  [ self ]
    (py/call-attr self "timeout"))
