(ns torch.cuda.profiler.cudaOutputMode
  ""
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce profiler (import-module "torch.cuda.profiler"))

(defn cudaOutputMode 
  ""
  [  ]
  (py/call-attr profiler "cudaOutputMode"  ))

(defn for-key 
  ""
  [ self key ]
  (py/call-attr self "for_key"  self key ))
