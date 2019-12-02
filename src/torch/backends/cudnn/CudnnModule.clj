(ns torch.backends.cudnn.CudnnModule
  ""
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce cudnn (import-module "torch.backends.cudnn"))

(defn CudnnModule 
  ""
  [ m name ]
  (py/call-attr m "CudnnModule"  m name ))
