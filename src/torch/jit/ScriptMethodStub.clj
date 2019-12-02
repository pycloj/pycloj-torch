(ns torch.jit.ScriptMethodStub
  "ScriptMethodStub(resolution_callback, def_, original_method)"
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce jit (import-module "torch.jit"))

(defn ScriptMethodStub 
  "ScriptMethodStub(resolution_callback, def_, original_method)"
  [ resolution_callback def_ original_method ]
  (py/call-attr jit "ScriptMethodStub"  resolution_callback def_ original_method ))

(defn def- 
  "Alias for field number 1"
  [ self ]
    (py/call-attr self "def_"))

(defn original-method 
  "Alias for field number 2"
  [ self ]
    (py/call-attr self "original_method"))

(defn resolution-callback 
  "Alias for field number 0"
  [ self ]
    (py/call-attr self "resolution_callback"))
