(ns torch.nn.modules.module.OrderedDict
  "Dictionary that remembers insertion order"
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce module (import-module "torch.nn.modules.module"))
