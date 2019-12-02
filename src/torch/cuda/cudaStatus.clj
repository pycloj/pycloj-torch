(ns torch.cuda.cudaStatus
  ""
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce cuda (import-module "torch.cuda"))

(defn cudaStatus 
  ""
  [  ]
  (py/call-attr cuda "cudaStatus"  ))
