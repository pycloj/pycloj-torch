(ns torch.onnx.ExportTypes
  ""
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce onnx (import-module "torch.onnx"))

(defn ExportTypes 
  ""
  [  ]
  (py/call-attr onnx "ExportTypes"  ))
