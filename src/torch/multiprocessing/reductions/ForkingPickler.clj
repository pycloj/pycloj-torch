(ns torch.multiprocessing.reductions.ForkingPickler
  "Pickler subclass used by multiprocessing."
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce reductions (import-module "torch.multiprocessing.reductions"))

(defn ForkingPickler 
  "Pickler subclass used by multiprocessing."
  [  ]
  (py/call-attr reductions "ForkingPickler"  ))
