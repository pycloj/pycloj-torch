(ns torch.nn.intrinsic.qat
  ""
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce qat (import-module "torch.nn.intrinsic.qat"))

(defn freeze-bn-stats 
  ""
  [ mod ]
  (py/call-attr qat "freeze_bn_stats"  mod ))

(defn update-bn-stats 
  ""
  [ mod ]
  (py/call-attr qat "update_bn_stats"  mod ))
