(ns torch.nn.utils.fusion
  ""
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce fusion (import-module "torch.nn.utils.fusion"))

(defn fuse-conv-bn-eval 
  ""
  [ conv bn ]
  (py/call-attr fusion "fuse_conv_bn_eval"  conv bn ))

(defn fuse-conv-bn-weights 
  ""
  [ conv_w conv_b bn_rm bn_rv bn_eps bn_w bn_b ]
  (py/call-attr fusion "fuse_conv_bn_weights"  conv_w conv_b bn_rm bn_rv bn_eps bn_w bn_b ))
