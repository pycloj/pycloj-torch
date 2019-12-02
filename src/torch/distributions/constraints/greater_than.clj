(ns torch.distributions.constraints.greater-than
  "
    Constrain to a real half line `(lower_bound, inf]`.
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

(defn greater-than 
  "
    Constrain to a real half line `(lower_bound, inf]`.
    "
  [ lower_bound ]
  (py/call-attr constraints "greater_than"  lower_bound ))

(defn check 
  ""
  [ self value ]
  (py/call-attr self "check"  self value ))
