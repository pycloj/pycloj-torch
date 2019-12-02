(ns torch.distributed.BroadcastOptions
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
