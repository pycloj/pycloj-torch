(ns torch.cuda.CudaError
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

(defn CudaError 
  ""
  [ code ]
  (py/call-attr cuda "CudaError"  code ))
