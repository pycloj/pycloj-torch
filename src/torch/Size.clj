(ns torch.Size
  ""
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce torch (import-module "torch"))

(defn Size 
  ""
  [ & {:keys [iterable]
       :or {iterable ()}} ]
  
   (py/call-attr-kw torch "Size" [] {:iterable iterable }))
