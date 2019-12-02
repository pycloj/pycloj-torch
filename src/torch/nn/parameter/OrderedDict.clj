(ns torch.nn.parameter.OrderedDict
  "Dictionary that remembers insertion order"
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce parameter (import-module "torch.nn.parameter"))
