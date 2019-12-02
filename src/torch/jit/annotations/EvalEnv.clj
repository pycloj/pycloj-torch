(ns torch.jit.annotations.EvalEnv
  ""
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce annotations (import-module "torch.jit.annotations"))

(defn EvalEnv 
  ""
  [ rcb ]
  (py/call-attr annotations "EvalEnv"  rcb ))
