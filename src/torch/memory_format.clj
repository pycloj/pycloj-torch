(ns torch.memory-format
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

(defn memory-format 
  ""
  [  ]
  (py/call-attr torch "memory_format"  ))
