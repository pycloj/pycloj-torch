(ns torch.distributions.constraints.interval
  "
    Constrain to a real interval `[lower_bound, upper_bound]`.
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

(defn interval 
  "
    Constrain to a real interval `[lower_bound, upper_bound]`.
    "
  [ lower_bound upper_bound ]
  (py/call-attr constraints "interval"  lower_bound upper_bound ))

(defn check 
  ""
  [ self value ]
  (py/call-attr self "check"  self value ))
