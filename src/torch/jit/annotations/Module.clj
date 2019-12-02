(ns torch.jit.annotations.Module
  ""
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce annotations (import-module "torch.jit.annotations"))

(defn Module 
  ""
  [ name members ]
  (py/call-attr annotations "Module"  name members ))
