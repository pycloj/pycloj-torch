(ns torch.nn.intrinsic.quantized.modules.conv-relu
  ""
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce conv-relu (import-module "torch.nn.intrinsic.quantized.modules.conv_relu"))

(defn fuse-conv-bn-weights 
  ""
  [ conv_w conv_b bn_rm bn_rv bn_eps bn_w bn_b ]
  (py/call-attr conv-relu "fuse_conv_bn_weights"  conv_w conv_b bn_rm bn_rv bn_eps bn_w bn_b ))
