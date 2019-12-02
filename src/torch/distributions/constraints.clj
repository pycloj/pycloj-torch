(ns torch.distributions.constraints
  "
The following constraints are implemented:

- ``constraints.boolean``
- ``constraints.cat``
- ``constraints.dependent``
- ``constraints.greater_than(lower_bound)``
- ``constraints.integer_interval(lower_bound, upper_bound)``
- ``constraints.interval(lower_bound, upper_bound)``
- ``constraints.lower_cholesky``
- ``constraints.lower_triangular``
- ``constraints.nonnegative_integer``
- ``constraints.positive``
- ``constraints.positive_definite``
- ``constraints.positive_integer``
- ``constraints.real``
- ``constraints.real_vector``
- ``constraints.simplex``
- ``constraints.stack``
- ``constraints.unit_interval``
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

(defn is-dependent 
  ""
  [ constraint ]
  (py/call-attr constraints "is_dependent"  constraint ))
