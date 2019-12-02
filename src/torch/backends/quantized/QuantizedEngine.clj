(ns torch.backends.quantized.QuantizedEngine
  ""
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce quantized (import-module "torch.backends.quantized"))

(defn QuantizedEngine 
  ""
  [ m name ]
  (py/call-attr m "QuantizedEngine"  m name ))
