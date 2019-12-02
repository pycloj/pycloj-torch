(ns torch.autograd.profiler.EnforceUnique
  "Raises an error if a key is seen more than once."
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce profiler (import-module "torch.autograd.profiler"))

(defn EnforceUnique 
  "Raises an error if a key is seen more than once."
  [  ]
  (py/call-attr profiler "EnforceUnique"  ))

(defn see 
  ""
  [ self  ]
  (py/call-attr self "see"  self  ))
