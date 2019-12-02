(ns torch.onnx.OperatorExportTypes
  "Members:

  ONNX

  ONNX_ATEN

  ONNX_ATEN_FALLBACK

  RAW"
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce onnx (import-module "torch.onnx"))

(defn name 
  "(self: handle) -> str
"
  [ self ]
    (py/call-attr self "name"))
