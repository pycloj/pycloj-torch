(ns torch.backends.mkl
  ""
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce mkl (import-module "torch.backends.mkl"))

(defn is-available 
  "Returns whether PyTorch is built with MKL support."
  [  ]
  (py/call-attr mkl "is_available"  ))
