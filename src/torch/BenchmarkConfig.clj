(ns torch.BenchmarkConfig
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

(defn num-calling-threads 
  ""
  [ self ]
    (py/call-attr self "num_calling_threads"))

(defn num-iters 
  ""
  [ self ]
    (py/call-attr self "num_iters"))

(defn num-warmup-iters 
  ""
  [ self ]
    (py/call-attr self "num_warmup_iters"))

(defn num-worker-threads 
  ""
  [ self ]
    (py/call-attr self "num_worker_threads"))
