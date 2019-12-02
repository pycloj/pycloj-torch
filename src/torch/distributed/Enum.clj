(ns torch.distributed.Enum
  "Generic enumeration.

    Derive from this class to define new enumerations.

    "
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce distributed (import-module "torch.distributed"))

(defn Enum 
  "Generic enumeration.

    Derive from this class to define new enumerations.

    "
  [value names module qualname type & {:keys [start]
                       :or {start 1}} ]
    (py/call-attr-kw distributed "Enum" [value names module qualname type] {:start start }))
