(ns torch.distributed.distributed-c10d.BroadcastOptions
  ""
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce distributed-c10d (import-module "torch.distributed.distributed_c10d"))

(defn rootRank 
  ""
  [ self ]
    (py/call-attr self "rootRank"))

(defn rootTensor 
  ""
  [ self ]
    (py/call-attr self "rootTensor"))

(defn timeout 
  ""
  [ self ]
    (py/call-attr self "timeout"))
