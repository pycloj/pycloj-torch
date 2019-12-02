(ns torch.jit.ScriptMeta
  ""
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce jit (import-module "torch.jit"))

(defn ScriptMeta 
  ""
  [ name bases attrs ]
  (py/call-attr jit "ScriptMeta"  name bases attrs ))
