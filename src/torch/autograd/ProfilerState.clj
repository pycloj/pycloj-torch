(ns torch.autograd.ProfilerState
  "Members:

  Disabled

  CPU

  CUDA

  NVTX"
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce autograd (import-module "torch.autograd"))

(defn name 
  "(self: handle) -> str
"
  [ self ]
    (py/call-attr self "name"))
