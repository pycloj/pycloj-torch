(ns torch.nn.quantized.modules.conv
  "Quantized convolution modules."
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce conv (import-module "torch.nn.quantized.modules.conv"))

(defn fuse-conv-bn-weights 
  ""
  [ conv_w conv_b bn_rm bn_rv bn_eps bn_w bn_b ]
  (py/call-attr conv "fuse_conv_bn_weights"  conv_w conv_b bn_rm bn_rv bn_eps bn_w bn_b ))
