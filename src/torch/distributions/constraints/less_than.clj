(ns torch.distributions.constraints.less-than
  "
    Constrain to a real half line `[-inf, upper_bound)`.
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

(defn less-than 
  "
    Constrain to a real half line `[-inf, upper_bound)`.
    "
  [ upper_bound ]
  (py/call-attr constraints "less_than"  upper_bound ))

(defn check 
  ""
  [ self value ]
  (py/call-attr self "check"  self value ))
