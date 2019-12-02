(ns torch.nn.parallel.distributed
  ""
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce distributed (import-module "torch.nn.parallel.distributed"))

(defn contextmanager 
  "@contextmanager decorator.

    Typical usage:

        @contextmanager
        def some_generator(<arguments>):
            <setup>
            try:
                yield <value>
            finally:
                <cleanup>

    This makes this:

        with some_generator(<arguments>) as <variable>:
            <body>

    equivalent to this:

        <setup>
        try:
            <variable> = <value>
            <body>
        finally:
            <cleanup>
    "
  [ func ]
  (py/call-attr distributed "contextmanager"  func ))

(defn gather 
  "
    Gathers tensors from different GPUs on a specified device
      (-1 means the CPU).
    "
  [outputs target_device & {:keys [dim]
                       :or {dim 0}} ]
    (py/call-attr-kw distributed "gather" [outputs target_device] {:dim dim }))

(defn parallel-apply 
  "Applies each `module` in :attr:`modules` in parallel on arguments
    contained in :attr:`inputs` (positional) and :attr:`kwargs_tup` (keyword)
    on each of :attr:`devices`.

    Args:
        modules (Module): modules to be parallelized
        inputs (tensor): inputs to the modules
        devices (list of int or torch.device): CUDA devices

    :attr:`modules`, :attr:`inputs`, :attr:`kwargs_tup` (if given), and
    :attr:`devices` (if given) should all have same length. Moreover, each
    element of :attr:`inputs` can either be a single object as the only argument
    to a module, or a collection of positional arguments.
    "
  [ modules inputs kwargs_tup devices ]
  (py/call-attr distributed "parallel_apply"  modules inputs kwargs_tup devices ))

(defn replicate 
  ""
  [network devices & {:keys [detach]
                       :or {detach false}} ]
    (py/call-attr-kw distributed "replicate" [network devices] {:detach detach }))

(defn scatter-kwargs 
  "Scatter with support for kwargs dictionary"
  [inputs kwargs target_gpus & {:keys [dim]
                       :or {dim 0}} ]
    (py/call-attr-kw distributed "scatter_kwargs" [inputs kwargs target_gpus] {:dim dim }))
