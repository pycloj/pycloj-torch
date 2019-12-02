(ns torch.cuda.nccl
  ""
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce nccl (import-module "torch.cuda.nccl"))

(defn all-gather 
  ""
  [ inputs outputs streams comms ]
  (py/call-attr nccl "all_gather"  inputs outputs streams comms ))

(defn all-reduce 
  ""
  [inputs outputs & {:keys [op streams comms]
                       :or {op 0}} ]
    (py/call-attr-kw nccl "all_reduce" [inputs outputs] {:op op :streams streams :comms comms }))

(defn broadcast 
  ""
  [inputs & {:keys [root streams comms]
                       :or {root 0}} ]
    (py/call-attr-kw nccl "broadcast" [inputs] {:root root :streams streams :comms comms }))

(defn init-rank 
  ""
  [ num_ranks uid rank ]
  (py/call-attr nccl "init_rank"  num_ranks uid rank ))

(defn is-available 
  ""
  [ tensors ]
  (py/call-attr nccl "is_available"  tensors ))

(defn reduce 
  ""
  [inputs outputs & {:keys [root op streams comms]
                       :or {root 0 op 0}} ]
    (py/call-attr-kw nccl "reduce" [inputs outputs] {:root root :op op :streams streams :comms comms }))

(defn reduce-scatter 
  ""
  [inputs outputs & {:keys [op streams comms]
                       :or {op 0}} ]
    (py/call-attr-kw nccl "reduce_scatter" [inputs outputs] {:op op :streams streams :comms comms }))

(defn unique-id 
  ""
  [  ]
  (py/call-attr nccl "unique_id"  ))

(defn version 
  ""
  [  ]
  (py/call-attr nccl "version"  ))
