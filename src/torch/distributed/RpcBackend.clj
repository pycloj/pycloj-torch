(ns torch.distributed.RpcBackend
  "An enumeration."
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce distributed (import-module "torch.distributed"))

(defn RpcBackend 
  "An enumeration."
  [value names module qualname type & {:keys [start]
                       :or {start 1}} ]
    (py/call-attr-kw distributed "RpcBackend" [value names module qualname type] {:start start }))
