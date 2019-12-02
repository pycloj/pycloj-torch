(ns torch.Use
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

(defn offset 
  ""
  [ self ]
    (py/call-attr self "offset"))

(defn user 
  ""
  [ self ]
    (py/call-attr self "user"))
