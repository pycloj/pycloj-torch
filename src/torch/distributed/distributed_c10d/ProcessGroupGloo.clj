(ns torch.distributed.distributed-c10d.ProcessGroupGloo
  ""
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce distributed-c10d (import-module "torch.distributed.distributed_c10d"))
