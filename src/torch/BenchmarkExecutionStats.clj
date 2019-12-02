(ns torch.BenchmarkExecutionStats
  ""
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce torch (import-module "torch"))

(defn latency-avg-ms 
  ""
  [ self ]
    (py/call-attr self "latency_avg_ms"))

(defn num-iters 
  ""
  [ self ]
    (py/call-attr self "num_iters"))
