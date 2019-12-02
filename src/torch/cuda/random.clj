(ns torch.cuda.random
  ""
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce random (import-module "torch.cuda.random"))

(defn current-device 
  "Returns the index of a currently selected device."
  [  ]
  (py/call-attr random "current_device"  ))

(defn device-count 
  "Returns the number of GPUs available."
  [  ]
  (py/call-attr random "device_count"  ))

(defn get-rng-state 
  "Returns the random number generator state of the specified GPU as a ByteTensor.

    Args:
        device (torch.device or int, optional): The device to return the RNG state of.
            Default: ``'cuda'`` (i.e., ``torch.device('cuda')``, the current CUDA device).

    .. warning::
        This function eagerly initializes CUDA.
    "
  [ & {:keys [device]
       :or {device "cuda"}} ]
  
   (py/call-attr-kw random "get_rng_state" [] {:device device }))

(defn get-rng-state-all 
  "Returns a tuple of ByteTensor representing the random number states of all devices."
  [  ]
  (py/call-attr random "get_rng_state_all"  ))

(defn initial-seed 
  "Returns the current random seed of the current GPU.

    .. warning::
        This function eagerly initializes CUDA.
    "
  [  ]
  (py/call-attr random "initial_seed"  ))

(defn manual-seed 
  "Sets the seed for generating random numbers for the current GPU.
    It's safe to call this function if CUDA is not available; in that
    case, it is silently ignored.

    Args:
        seed (int): The desired seed.

    .. warning::
        If you are working with a multi-GPU model, this function is insufficient
        to get determinism.  To seed all GPUs, use :func:`manual_seed_all`.
    "
  [ seed ]
  (py/call-attr random "manual_seed"  seed ))

(defn manual-seed-all 
  "Sets the seed for generating random numbers on all GPUs.
    It's safe to call this function if CUDA is not available; in that
    case, it is silently ignored.

    Args:
        seed (int): The desired seed.
    "
  [ seed ]
  (py/call-attr random "manual_seed_all"  seed ))

(defn seed 
  "Sets the seed for generating random numbers to a random number for the current GPU.
    It's safe to call this function if CUDA is not available; in that
    case, it is silently ignored.

    .. warning::
        If you are working with a multi-GPU model, this function will only initialize
        the seed on one GPU.  To initialize all GPUs, use :func:`seed_all`.
    "
  [  ]
  (py/call-attr random "seed"  ))

(defn seed-all 
  "Sets the seed for generating random numbers to a random number on all GPUs.
    It's safe to call this function if CUDA is not available; in that
    case, it is silently ignored.
    "
  [  ]
  (py/call-attr random "seed_all"  ))

(defn set-rng-state 
  "Sets the random number generator state of the specified GPU.

    Args:
        new_state (torch.ByteTensor): The desired state
        device (torch.device or int, optional): The device to set the RNG state.
            Default: ``'cuda'`` (i.e., ``torch.device('cuda')``, the current CUDA device).
    "
  [new_state & {:keys [device]
                       :or {device "cuda"}} ]
    (py/call-attr-kw random "set_rng_state" [new_state] {:device device }))

(defn set-rng-state-all 
  "Sets the random number generator state of all devices.

    Args:
        new_state (tuple of torch.ByteTensor): The desired state for each device"
  [ new_states ]
  (py/call-attr random "set_rng_state_all"  new_states ))
