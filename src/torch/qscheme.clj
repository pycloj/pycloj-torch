(ns torch.qscheme
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

(defn qscheme 
  ""
  [  ]
  (py/call-attr torch "qscheme"  ))
