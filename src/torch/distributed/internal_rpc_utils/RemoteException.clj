(ns torch.distributed.internal-rpc-utils.RemoteException
  "RemoteException(msg,)"
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce internal-rpc-utils (import-module "torch.distributed.internal_rpc_utils"))

(defn RemoteException 
  "RemoteException(msg,)"
  [ msg ]
  (py/call-attr internal-rpc-utils "RemoteException"  msg ))

(defn msg 
  "Alias for field number 0"
  [ self ]
    (py/call-attr self "msg"))
