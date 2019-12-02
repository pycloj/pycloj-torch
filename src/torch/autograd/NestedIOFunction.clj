(ns torch.autograd.NestedIOFunction
  ""
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce autograd (import-module "torch.autograd"))

(defn backward 
  ""
  [ self  ]
  (py/call-attr self "backward"  self  ))

(defn backward-extended 
  ""
  [ self  ]
  (py/call-attr self "backward_extended"  self  ))

(defn forward 
  ""
  [ self  ]
  (py/call-attr self "forward"  self  ))

(defn forward-extended 
  ""
  [ self  ]
  (py/call-attr self "forward_extended"  self  ))

(defn mark-dirty 
  ""
  [ self  ]
  (py/call-attr self "mark_dirty"  self  ))

(defn mark-non-differentiable 
  ""
  [ self  ]
  (py/call-attr self "mark_non_differentiable"  self  ))

(defn mark-shared-storage 
  ""
  [ self  ]
  (py/call-attr self "mark_shared_storage"  self  ))

(defn save-for-backward 
  ""
  [ self  ]
  (py/call-attr self "save_for_backward"  self  ))

(defn saved-tensors 
  ""
  [ self ]
    (py/call-attr self "saved_tensors"))
