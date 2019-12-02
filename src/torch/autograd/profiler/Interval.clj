(ns torch.autograd.profiler.Interval
  ""
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce profiler (import-module "torch.autograd.profiler"))

(defn Interval 
  ""
  [ start end ]
  (py/call-attr profiler "Interval"  start end ))

(defn elapsed-us 
  ""
  [ self  ]
  (py/call-attr self "elapsed_us"  self  ))
