(ns torch.backends.openmp
  ""
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce openmp (import-module "torch.backends.openmp"))

(defn is-available 
  "Returns whether PyTorch is built with OpenMP support."
  [  ]
  (py/call-attr openmp "is_available"  ))
