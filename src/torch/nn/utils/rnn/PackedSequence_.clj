(ns torch.nn.utils.rnn.PackedSequence-
  "PackedSequence(data, batch_sizes, sorted_indices, unsorted_indices)"
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce rnn (import-module "torch.nn.utils.rnn"))

(defn PackedSequence- 
  "PackedSequence(data, batch_sizes, sorted_indices, unsorted_indices)"
  [ data batch_sizes sorted_indices unsorted_indices ]
  (py/call-attr rnn "PackedSequence_"  data batch_sizes sorted_indices unsorted_indices ))

(defn batch-sizes 
  "Alias for field number 1"
  [ self ]
    (py/call-attr self "batch_sizes"))

(defn data 
  "Alias for field number 0"
  [ self ]
    (py/call-attr self "data"))

(defn sorted-indices 
  "Alias for field number 2"
  [ self ]
    (py/call-attr self "sorted_indices"))

(defn unsorted-indices 
  "Alias for field number 3"
  [ self ]
    (py/call-attr self "unsorted_indices"))
