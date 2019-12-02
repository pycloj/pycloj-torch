(ns torch.autograd.profiler.FunctionEvent
  "Profiling information about a single function."
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce profiler (import-module "torch.autograd.profiler"))

(defn FunctionEvent 
  "Profiling information about a single function."
  [ id name thread cpu_start cpu_end input_shapes ]
  (py/call-attr profiler "FunctionEvent"  id name thread cpu_start cpu_end input_shapes ))

(defn append-cpu-child 
  "Append a CPU child of type FunctionEvent.

        One is supposed to append only dirrect children to the event to have
        correct self cpu time being reported.
        "
  [ self child ]
  (py/call-attr self "append_cpu_child"  self child ))

(defn append-kernel 
  ""
  [ self name device start end ]
  (py/call-attr self "append_kernel"  self name device start end ))

(defn cpu-time 
  ""
  [ self ]
    (py/call-attr self "cpu_time"))

(defn cpu-time-str 
  ""
  [ self ]
    (py/call-attr self "cpu_time_str"))

(defn cpu-time-total 
  ""
  [ self ]
    (py/call-attr self "cpu_time_total"))

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

(defn cuda-time-total 
  ""
  [ self ]
    (py/call-attr self "cuda_time_total"))

(defn cuda-time-total-str 
  ""
  [ self ]
    (py/call-attr self "cuda_time_total_str"))

(defn key 
  ""
  [ self ]
    (py/call-attr self "key"))

(defn self-cpu-time-total 
  ""
  [ self ]
    (py/call-attr self "self_cpu_time_total"))

(defn self-cpu-time-total-str 
  ""
  [ self ]
    (py/call-attr self "self_cpu_time_total_str"))
