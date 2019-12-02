(ns torch.distributed.WorkerId
  ""
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce distributed (import-module "torch.distributed"))

(defn id 
  ""
  [ self ]
    (py/call-attr self "id"))

(defn name 
  ""
  [ self ]
    (py/call-attr self "name"))
