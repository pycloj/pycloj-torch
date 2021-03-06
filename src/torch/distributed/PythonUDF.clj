(ns torch.distributed.PythonUDF
  "PythonUDF(func, args, kwargs)"
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce distributed (import-module "torch.distributed"))

(defn PythonUDF 
  "PythonUDF(func, args, kwargs)"
  [ func args kwargs ]
  (py/call-attr distributed "PythonUDF"  func args kwargs ))

(defn args 
  "Alias for field number 1"
  [ self ]
    (py/call-attr self "args"))

(defn func 
  "Alias for field number 0"
  [ self ]
    (py/call-attr self "func"))

(defn kwargs 
  "Alias for field number 2"
  [ self ]
    (py/call-attr self "kwargs"))
