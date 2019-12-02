(ns torch.jit.frontend.UnsupportedNodeError
  ""
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce frontend (import-module "torch.jit.frontend"))

(defn UnsupportedNodeError 
  ""
  [ ctx offending_node ]
  (py/call-attr frontend "UnsupportedNodeError"  ctx offending_node ))
