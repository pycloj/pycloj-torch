(ns torch.backends.cudnn.CuDNNError
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

(defn CuDNNError 
  ""
  [ status ]
  (py/call-attr m "CuDNNError"  status ))
