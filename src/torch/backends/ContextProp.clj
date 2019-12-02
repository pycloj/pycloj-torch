(ns torch.backends.ContextProp
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

(defn ContextProp 
  ""
  [ getter setter ]
  (py/call-attr backends "ContextProp"  getter setter ))
