(ns torch.backends.cudnn.DropoutDescriptor
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

(defn DropoutDescriptor 
  ""
  [ handle dropout seed ]
  (py/call-attr m "DropoutDescriptor"  handle dropout seed ))

(defn set-dropout 
  ""
  [ self dropout seed ]
  (py/call-attr self "set_dropout"  self dropout seed ))
