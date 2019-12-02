(ns torch.distributions.constraints.Constraint
  "
    Abstract base class for constraints.

    A constraint object represents a region over which a variable is valid,
    e.g. within which a variable can be optimized.
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

(defn Constraint 
  "
    Abstract base class for constraints.

    A constraint object represents a region over which a variable is valid,
    e.g. within which a variable can be optimized.
    "
  [  ]
  (py/call-attr constraints "Constraint"  ))

(defn check 
  "
        Returns a byte tensor of `sample_shape + batch_shape` indicating
        whether each event in value satisfies this constraint.
        "
  [ self value ]
  (py/call-attr self "check"  self value ))
