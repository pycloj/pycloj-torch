(ns torch.backends.cudnn.CuDNNHandle
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

(defn CuDNNHandle 
  ""
  [  ]
  (py/call-attr m "CuDNNHandle"  ))
