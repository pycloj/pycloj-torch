(ns torch.autograd.profiler.FormattedTimesMixin
  "Helpers for FunctionEvent and FunctionEventAvg.

    The subclass should define `*_time_total` and `count` attributes.
    "
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce profiler (import-module "torch.autograd.profiler"))

(defn FormattedTimesMixin 
  "Helpers for FunctionEvent and FunctionEventAvg.

    The subclass should define `*_time_total` and `count` attributes.
    "
  [  ]
  (py/call-attr profiler "FormattedTimesMixin"  ))

(defn cpu-time 
  ""
  [ self ]
    (py/call-attr self "cpu_time"))

(defn cpu-time-str 
  ""
  [ self ]
    (py/call-attr self "cpu_time_str"))

(defn cpu-time-total-str 
  ""
  [ self ]
    (py/call-attr self "cpu_time_total_str"))

(defn cuda-time 
  ""
  [ self ]
    (py/call-attr self "cuda_time"))

(defn cuda-time-str 
  ""
  [ self ]
    (py/call-attr self "cuda_time_str"))

(defn cuda-time-total-str 
  ""
  [ self ]
    (py/call-attr self "cuda_time_total_str"))

(defn self-cpu-time-total-str 
  ""
  [ self ]
    (py/call-attr self "self_cpu_time_total_str"))
