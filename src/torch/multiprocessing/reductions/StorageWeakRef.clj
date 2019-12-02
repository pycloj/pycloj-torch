(ns torch.multiprocessing.reductions.StorageWeakRef
  "A weak reference to a Storage.

    The cdata member is a Python number containing the integer representation of
    the Storage pointer."
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce reductions (import-module "torch.multiprocessing.reductions"))

(defn StorageWeakRef 
  "A weak reference to a Storage.

    The cdata member is a Python number containing the integer representation of
    the Storage pointer."
  [ storage ]
  (py/call-attr reductions "StorageWeakRef"  storage ))

(defn expired 
  ""
  [ self  ]
  (py/call-attr self "expired"  self  ))
