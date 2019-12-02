(ns torch.distributions.constraints.integer-interval
  "
    Constrain to an integer interval `[lower_bound, upper_bound]`.
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

(defn integer-interval 
  "
    Constrain to an integer interval `[lower_bound, upper_bound]`.
    "
  [ lower_bound upper_bound ]
  (py/call-attr constraints "integer_interval"  lower_bound upper_bound ))

(defn check 
  ""
  [ self value ]
  (py/call-attr self "check"  self value ))
