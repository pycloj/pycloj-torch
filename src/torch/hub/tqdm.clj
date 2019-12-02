(ns torch.hub.tqdm
  ""
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce hub (import-module "torch.hub"))

(defn tqdm 
  ""
  [total & {:keys [disable unit unit_scale unit_divisor]
                       :or {disable false}} ]
    (py/call-attr-kw hub "tqdm" [total] {:disable disable :unit unit :unit_scale unit_scale :unit_divisor unit_divisor }))

(defn update 
  ""
  [ self n ]
  (py/call-attr self "update"  self n ))
