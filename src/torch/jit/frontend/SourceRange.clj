(ns torch.jit.frontend.SourceRange
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

(defn end 
  ""
  [ self ]
    (py/call-attr self "end"))

(defn start 
  ""
  [ self ]
    (py/call-attr self "start"))
