(ns torch.backends.cudnn.ContextProp
  ""
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce cudnn (import-module "torch.backends.cudnn"))

(defn ContextProp 
  ""
  [ getter setter ]
  (py/call-attr m "ContextProp"  getter setter ))
