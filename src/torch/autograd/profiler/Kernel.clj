(ns torch.autograd.profiler.Kernel
  "Kernel(name, device, interval)"
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce profiler (import-module "torch.autograd.profiler"))

(defn Kernel 
  "Kernel(name, device, interval)"
  [ name device interval ]
  (py/call-attr profiler "Kernel"  name device interval ))

(defn device 
  "Alias for field number 1"
  [ self ]
    (py/call-attr self "device"))

(defn interval 
  "Alias for field number 2"
  [ self ]
    (py/call-attr self "interval"))

(defn name 
  "Alias for field number 0"
  [ self ]
    (py/call-attr self "name"))
