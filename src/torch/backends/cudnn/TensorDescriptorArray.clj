(ns torch.backends.cudnn.TensorDescriptorArray
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

(defn TensorDescriptorArray 
  ""
  [ N ]
  (py/call-attr m "TensorDescriptorArray"  N ))

(defn set-all 
  ""
  [ self tensor ]
  (py/call-attr self "set_all"  self tensor ))

(defn set-raw 
  ""
  [ self i _type _ndim _size _stride ]
  (py/call-attr self "set_raw"  self i _type _ndim _size _stride ))
