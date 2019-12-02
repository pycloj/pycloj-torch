(ns torch.distributions.multinomial.Number
  "All numbers inherit from this class.

    If you just want to check if an argument x is a number, without
    caring what kind, use isinstance(x, Number).
    "
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce multinomial (import-module "torch.distributions.multinomial"))

(defn Number 
  "All numbers inherit from this class.

    If you just want to check if an argument x is a number, without
    caring what kind, use isinstance(x, Number).
    "
  [  ]
  (py/call-attr multinomial "Number"  ))
