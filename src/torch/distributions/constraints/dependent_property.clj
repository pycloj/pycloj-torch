(ns torch.distributions.constraints.dependent-property
  "
    Decorator that extends @property to act like a `Dependent` constraint when
    called on a class and act like a property when called on an object.

    Example::

        class Uniform(Distribution):
            def __init__(self, low, high):
                self.low = low
                self.high = high
            @constraints.dependent_property
            def support(self):
                return constraints.interval(self.low, self.high)
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

(defn dependent-property 
  "
    Decorator that extends @property to act like a `Dependent` constraint when
    called on a class and act like a property when called on an object.

    Example::

        class Uniform(Distribution):
            def __init__(self, low, high):
                self.low = low
                self.high = high
            @constraints.dependent_property
            def support(self):
                return constraints.interval(self.low, self.high)
    "
  [ fget fset fdel doc ]
  (py/call-attr constraints "dependent_property"  fget fset fdel doc ))

(defn check 
  ""
  [ self x ]
  (py/call-attr self "check"  self x ))
