(ns torch.distributions.multivariate-normal.lazy-property
  "
    Used as a decorator for lazy loading of class attributes. This uses a
    non-data descriptor that calls the wrapped method to compute the property on
    first call; thereafter replacing the wrapped method into an instance
    attribute.
    "
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce multivariate-normal (import-module "torch.distributions.multivariate_normal"))

(defn lazy-property 
  "
    Used as a decorator for lazy loading of class attributes. This uses a
    non-data descriptor that calls the wrapped method to compute the property on
    first call; thereafter replacing the wrapped method into an instance
    attribute.
    "
  [ wrapped ]
  (py/call-attr multivariate-normal "lazy_property"  wrapped ))
