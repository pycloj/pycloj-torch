(ns torch.multiprocessing.SpawnContext
  ""
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce multiprocessing (import-module "torch.multiprocessing"))

(defn SpawnContext 
  ""
  [ processes error_queues ]
  (py/call-attr multiprocessing "SpawnContext"  processes error_queues ))

(defn join 
  "
        Tries to join one or more processes in this spawn context.
        If one of them exited with a non-zero exit status, this function
        kills the remaining processes and raises an exception with the cause
        of the first process exiting.

        Returns ``True`` if all processes have been joined successfully,
        ``False`` if there are more processes that need to be joined.

        Arguments:
            timeout (float): Wait this long before giving up on waiting.
        "
  [ self timeout ]
  (py/call-attr self "join"  self timeout ))

(defn pids 
  ""
  [ self  ]
  (py/call-attr self "pids"  self  ))
