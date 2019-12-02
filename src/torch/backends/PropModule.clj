(ns torch.backends.PropModule
  ""
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce backends (import-module "torch.backends"))

(defn PropModule 
  ""
  [ m name ]
  (py/call-attr backends "PropModule"  m name ))
