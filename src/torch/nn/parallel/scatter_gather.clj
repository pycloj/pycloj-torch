(ns torch.nn.parallel.scatter-gather
  ""
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce scatter-gather (import-module "torch.nn.parallel.scatter_gather"))

(defn gather 
  "
    Gathers tensors from different GPUs on a specified device
      (-1 means the CPU).
    "
  [outputs target_device & {:keys [dim]
                       :or {dim 0}} ]
    (py/call-attr-kw scatter-gather "gather" [outputs target_device] {:dim dim }))

(defn scatter 
  "
    Slices tensors into approximately equal chunks and
    distributes them across given GPUs. Duplicates
    references to objects that are not tensors.
    "
  [inputs target_gpus & {:keys [dim]
                       :or {dim 0}} ]
    (py/call-attr-kw scatter-gather "scatter" [inputs target_gpus] {:dim dim }))

(defn scatter-kwargs 
  "Scatter with support for kwargs dictionary"
  [inputs kwargs target_gpus & {:keys [dim]
                       :or {dim 0}} ]
    (py/call-attr-kw scatter-gather "scatter_kwargs" [inputs kwargs target_gpus] {:dim dim }))
