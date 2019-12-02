(ns torch.nn.intrinsic.qat.modules.conv-fused
  ""
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce conv-fused (import-module "torch.nn.intrinsic.qat.modules.conv_fused"))

(defn freeze-bn-stats 
  ""
  [ mod ]
  (py/call-attr conv-fused "freeze_bn_stats"  mod ))

(defn update-bn-stats 
  ""
  [ mod ]
  (py/call-attr conv-fused "update_bn_stats"  mod ))
