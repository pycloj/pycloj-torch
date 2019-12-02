(ns torch.jit.TracerWarning
  ""
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce jit (import-module "torch.jit"))

(defn ignore-lib-warnings 
  ""
  [ self  ]
  (py/call-attr self "ignore_lib_warnings"  self  ))
