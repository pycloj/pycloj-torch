(ns torch.nn.quantized.dynamic.modules.rnn
  ""
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce rnn (import-module "torch.nn.quantized.dynamic.modules.rnn"))

(defn apply-permutation 
  ""
  [tensor permutation & {:keys [dim]
                       :or {dim 1}} ]
    (py/call-attr-kw rnn "apply_permutation" [tensor permutation] {:dim dim }))
