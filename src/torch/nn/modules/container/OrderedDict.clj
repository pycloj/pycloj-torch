(ns torch.nn.modules.container.OrderedDict
  "Dictionary that remembers insertion order"
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce container (import-module "torch.nn.modules.container"))
