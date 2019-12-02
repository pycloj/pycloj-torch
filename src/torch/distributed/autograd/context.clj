(ns torch.distributed.autograd.context
  "
    Autograd context object to wrap forward and backward passes when using
    distributed autograd. The context_id generated in the 'with' is required
    to uniquely identify a distributed autograd pass on all workers. Each
    worker stores metadata associated with this context_id, which is required
    to correctly execute a distributed autograd pass.

    This is only needed in the \"FAST\" mode for distributed autograd, where we
    assume all RPC communication is would also be part of the backward pass.

    Example::
        >> import torch.distributed.autograd as dist_autograd
        >> with dist_autograd.context() as context_id:
        >>      forward pass...
        >>      backward pass...
        >>      optimizer step...
    "
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce autograd (import-module "torch.distributed.autograd"))

(defn context 
  "
    Autograd context object to wrap forward and backward passes when using
    distributed autograd. The context_id generated in the 'with' is required
    to uniquely identify a distributed autograd pass on all workers. Each
    worker stores metadata associated with this context_id, which is required
    to correctly execute a distributed autograd pass.

    This is only needed in the \"FAST\" mode for distributed autograd, where we
    assume all RPC communication is would also be part of the backward pass.

    Example::
        >> import torch.distributed.autograd as dist_autograd
        >> with dist_autograd.context() as context_id:
        >>      forward pass...
        >>      backward pass...
        >>      optimizer step...
    "
  [  ]
  (py/call-attr autograd "context"  ))
