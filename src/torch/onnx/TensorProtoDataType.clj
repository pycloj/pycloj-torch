(ns torch.onnx.TensorProtoDataType
  "Members:

  UNDEFINED

  FLOAT

  UINT8

  INT8

  UINT16

  INT16

  INT32

  INT64

  STRING

  BOOL

  FLOAT16

  DOUBLE

  UINT32

  UINT64

  COMPLEX64

  COMPLEX128"
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
