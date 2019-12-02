(ns torch.nn.intrinsic.qat.modules
  ""
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce modules (import-module "torch.nn.intrinsic.qat.modules"))

(defn freeze-bn-stats 
  ""
  [ mod ]
  (py/call-attr modules "freeze_bn_stats"  mod ))

(defn update-bn-stats 
  ""
  [ mod ]
  (py/call-attr modules "update_bn_stats"  mod ))
