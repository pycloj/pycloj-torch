(ns torch.autograd.profiler.FunctionEventAvg
  "Used to average stats over multiple FunctionEvent objects."
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce profiler (import-module "torch.autograd.profiler"))

(defn FunctionEventAvg 
  "Used to average stats over multiple FunctionEvent objects."
  [  ]
  (py/call-attr profiler "FunctionEventAvg"  ))

(defn add 
  ""
  [self other & {:keys [group_by_input_shapes]
                       :or {group_by_input_shapes false}} ]
    (py/call-attr-kw self "add" [other] {:group_by_input_shapes group_by_input_shapes }))

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
