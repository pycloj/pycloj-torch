(ns torch.multiprocessing.reductions.SharedCache
  "dictionary from multiprocessing handles to StorageWeakRef"
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce reductions (import-module "torch.multiprocessing.reductions"))

(defn SharedCache 
  "dictionary from multiprocessing handles to StorageWeakRef"
  [  ]
  (py/call-attr reductions "SharedCache"  ))

(defn free-dead-references 
  ""
  [ self  ]
  (py/call-attr self "free_dead_references"  self  ))
