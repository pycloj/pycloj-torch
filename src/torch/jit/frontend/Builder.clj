(ns torch.jit.frontend.Builder
  ""
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce frontend (import-module "torch.jit.frontend"))

(defn Builder 
  ""
  [  ]
  (py/call-attr frontend "Builder"  ))
