(ns torch.jit.TracingCheckError
  ""
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce jit (import-module "torch.jit"))

(defn TracingCheckError 
  ""
  [ graph_diff_error tensor_compare_error extra_msg ]
  (py/call-attr jit "TracingCheckError"  graph_diff_error tensor_compare_error extra_msg ))
