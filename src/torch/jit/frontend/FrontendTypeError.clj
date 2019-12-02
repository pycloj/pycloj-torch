(ns torch.jit.frontend.FrontendTypeError
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

(defn FrontendTypeError 
  ""
  [ source_range msg ]
  (py/call-attr frontend "FrontendTypeError"  source_range msg ))
