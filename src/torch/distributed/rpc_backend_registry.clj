(ns torch.distributed.rpc-backend-registry
  ""
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce rpc-backend-registry (import-module "torch.distributed.rpc_backend_registry"))

(defn init-rpc-backend 
  ""
  [ backend_name ]
  (py/call-attr rpc-backend-registry "init_rpc_backend"  backend_name ))

(defn is-rpc-backend-registered 
  ""
  [ backend_name ]
  (py/call-attr rpc-backend-registry "is_rpc_backend_registered"  backend_name ))

(defn register-rpc-backend 
  "Registers a new rpc backend.

    Arguments:
        backend (str): backend string to identify the handler.
        handler (function): Handler that is invoked when the
            `_init_rpc()` function is called with a backend.
             This returns the agent.
    "
  [ backend_name init_rpc_backend_handler ]
  (py/call-attr rpc-backend-registry "register_rpc_backend"  backend_name init_rpc_backend_handler ))
