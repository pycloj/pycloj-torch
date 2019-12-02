(ns torch.nn.parallel
  ""
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce parallel (import-module "torch.nn.parallel"))

(defn DistributedDataParallelCPU 
  ""
  [  ]
  (py/call-attr parallel "DistributedDataParallelCPU"  ))

(defn data-parallel 
  "Evaluates module(input) in parallel across the GPUs given in device_ids.

    This is the functional version of the DataParallel module.

    Args:
        module (Module): the module to evaluate in parallel
        inputs (Tensor): inputs to the module
        device_ids (list of int or torch.device): GPU ids on which to replicate module
        output_device (list of int or torch.device): GPU location of the output  Use -1 to indicate the CPU.
            (default: device_ids[0])
    Returns:
        a Tensor containing the result of module(input) located on
        output_device
    "
  [module inputs device_ids output_device & {:keys [dim module_kwargs]
                       :or {dim 0}} ]
    (py/call-attr-kw parallel "data_parallel" [module inputs device_ids output_device] {:dim dim :module_kwargs module_kwargs }))

(defn gather 
  "
    Gathers tensors from different GPUs on a specified device
      (-1 means the CPU).
    "
  [outputs target_device & {:keys [dim]
                       :or {dim 0}} ]
    (py/call-attr-kw parallel "gather" [outputs target_device] {:dim dim }))

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
  (py/call-attr parallel "parallel_apply"  modules inputs kwargs_tup devices ))

(defn replicate 
  ""
  [network devices & {:keys [detach]
                       :or {detach false}} ]
    (py/call-attr-kw parallel "replicate" [network devices] {:detach detach }))

(defn scatter 
  "
    Slices tensors into approximately equal chunks and
    distributes them across given GPUs. Duplicates
    references to objects that are not tensors.
    "
  [inputs target_gpus & {:keys [dim]
                       :or {dim 0}} ]
    (py/call-attr-kw parallel "scatter" [inputs target_gpus] {:dim dim }))
