(ns torch.distributions.constraints.half-open-interval
  "
    Constrain to a real interval `[lower_bound, upper_bound)`.
    "
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce constraints (import-module "torch.distributions.constraints"))

(defn half-open-interval 
  "
    Constrain to a real interval `[lower_bound, upper_bound)`.
    "
  [ lower_bound upper_bound ]
  (py/call-attr constraints "half_open_interval"  lower_bound upper_bound ))

(defn check 
  ""
  [ self value ]
  (py/call-attr self "check"  self value ))
