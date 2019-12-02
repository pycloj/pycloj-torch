(ns torch.jit.frontend.FrontendError
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

(defn FrontendError 
  ""
  [ source_range msg ]
  (py/call-attr frontend "FrontendError"  source_range msg ))
