(ns torch.backends.cudnn.RNNDescriptor
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

(defn RNNDescriptor 
  ""
  [ handle hidden_size num_layers dropout_desc input_mode bidirectional mode datatype ]
  (py/call-attr m "RNNDescriptor"  handle hidden_size num_layers dropout_desc input_mode bidirectional mode datatype ))
