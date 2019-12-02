(ns torch.cuda
  "
This package adds support for CUDA tensor types, that implement the same
function as CPU tensors, but they utilize GPUs for computation.

It is lazily initialized, so you can always import it, and use
:func:`is_available()` to determine if your system supports CUDA.

:ref:`cuda-semantics` has more details about working with CUDA.
"
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce cuda (import-module "torch.cuda"))

(defn check-error 
  ""
  [ res ]
  (py/call-attr cuda "check_error"  res ))

(defn cudart 
  ""
  [  ]
  (py/call-attr cuda "cudart"  ))

(defn current-blas-handle 
  "Returns cublasHandle_t pointer to current cuBLAS handle"
  [  ]
  (py/call-attr cuda "current_blas_handle"  ))

(defn current-device 
  "Returns the index of a currently selected device."
  [  ]
  (py/call-attr cuda "current_device"  ))

(defn current-stream 
  "Returns the currently selected :class:`Stream` for a given device.

    Arguments:
        device (torch.device or int, optional): selected device. Returns
            the currently selected :class:`Stream` for the current device, given
            by :func:`~torch.cuda.current_device`, if :attr:`device` is ``None``
            (default).
    "
  [ device ]
  (py/call-attr cuda "current_stream"  device ))

(defn default-stream 
  "Returns the default :class:`Stream` for a given device.

    Arguments:
        device (torch.device or int, optional): selected device. Returns
            the default :class:`Stream` for the current device, given by
            :func:`~torch.cuda.current_device`, if :attr:`device` is ``None``
            (default).
    "
  [ device ]
  (py/call-attr cuda "default_stream"  device ))

(defn device-count 
  "Returns the number of GPUs available."
  [  ]
  (py/call-attr cuda "device_count"  ))

(defn empty-cache 
  "Releases all unoccupied cached memory currently held by the caching
    allocator so that those can be used in other GPU application and visible in
    `nvidia-smi`.

    .. note::
        :func:`~torch.cuda.empty_cache` doesn't increase the amount of GPU
        memory available for PyTorch. See :ref:`cuda-memory-management` for
        more details about GPU memory management.
    "
  [  ]
  (py/call-attr cuda "empty_cache"  ))

(defn find-cuda-windows-lib 
  ""
  [  ]
  (py/call-attr cuda "find_cuda_windows_lib"  ))

(defn get-device-capability 
  "Gets the cuda capability of a device.

    Arguments:
        device (torch.device or int, optional): device for which to return the
            device capability. This function is a no-op if this argument is
            a negative integer. It uses the current device, given by
            :func:`~torch.cuda.current_device`, if :attr:`device` is ``None``
            (default).

    Returns:
        tuple(int, int): the major and minor cuda capability of the device
    "
  [ device ]
  (py/call-attr cuda "get_device_capability"  device ))

(defn get-device-name 
  "Gets the name of a device.

    Arguments:
        device (torch.device or int, optional): device for which to return the
            name. This function is a no-op if this argument is a negative
            integer. It uses the current device, given by :func:`~torch.cuda.current_device`,
            if :attr:`device` is ``None`` (default).
    "
  [ device ]
  (py/call-attr cuda "get_device_name"  device ))

(defn get-device-properties 
  ""
  [ device ]
  (py/call-attr cuda "get_device_properties"  device ))

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
  
   (py/call-attr-kw cuda "get_rng_state" [] {:device device }))

(defn get-rng-state-all 
  "Returns a tuple of ByteTensor representing the random number states of all devices."
  [  ]
  (py/call-attr cuda "get_rng_state_all"  ))

(defn init 
  "Initialize PyTorch's CUDA state.  You may need to call
    this explicitly if you are interacting with PyTorch via
    its C API, as Python bindings for CUDA functionality will not
    be until this initialization takes place.  Ordinary users
    should not need this, as all of PyTorch's CUDA methods
    automatically initialize CUDA state on-demand.

    Does nothing if the CUDA state is already initialized.
    "
  [  ]
  (py/call-attr cuda "init"  ))

(defn initial-seed 
  "Returns the current random seed of the current GPU.

    .. warning::
        This function eagerly initializes CUDA.
    "
  [  ]
  (py/call-attr cuda "initial_seed"  ))

(defn ipc-collect 
  "Force collects GPU memory after it has been released by CUDA IPC.

    .. note::
        Checks if any sent CUDA tensors could be cleaned from the memory. Force
        closes shared memory file used for reference counting if there is no
        active counters. Useful when the producer process stopped actively sending
        tensors and want to release unused memory.
    "
  [  ]
  (py/call-attr cuda "ipc_collect"  ))

(defn is-available 
  "Returns a bool indicating if CUDA is currently available."
  [  ]
  (py/call-attr cuda "is_available"  ))

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
  (py/call-attr cuda "manual_seed"  seed ))

(defn manual-seed-all 
  "Sets the seed for generating random numbers on all GPUs.
    It's safe to call this function if CUDA is not available; in that
    case, it is silently ignored.

    Args:
        seed (int): The desired seed.
    "
  [ seed ]
  (py/call-attr cuda "manual_seed_all"  seed ))

(defn max-memory-allocated 
  "Returns the maximum GPU memory occupied by tensors in bytes for a given
    device.

    By default, this returns the peak allocated memory since the beginning of
    this program. :func:`~torch.cuda.reset_max_memory_allocated` can be used to
    reset the starting point in tracking this metric. For example, these two
    functions can measure the peak allocated memory usage of each iteration in a
    training loop.

    Arguments:
        device (torch.device or int, optional): selected device. Returns
            statistic for the current device, given by :func:`~torch.cuda.current_device`,
            if :attr:`device` is ``None`` (default).

    .. note::
        See :ref:`cuda-memory-management` for more details about GPU memory
        management.
    "
  [ device ]
  (py/call-attr cuda "max_memory_allocated"  device ))

(defn max-memory-cached 
  "Returns the maximum GPU memory managed by the caching allocator in bytes
    for a given device.

    By default, this returns the peak cached memory since the beginning of this
    program. :func:`~torch.cuda.reset_max_memory_cached` can be used to reset
    the starting point in tracking this metric. For example, these two functions
    can measure the peak cached memory amount of each iteration in a training
    loop.

    Arguments:
        device (torch.device or int, optional): selected device. Returns
            statistic for the current device, given by :func:`~torch.cuda.current_device`,
            if :attr:`device` is ``None`` (default).

    .. note::
        See :ref:`cuda-memory-management` for more details about GPU memory
        management.
    "
  [ device ]
  (py/call-attr cuda "max_memory_cached"  device ))

(defn memory-allocated 
  "Returns the current GPU memory occupied by tensors in bytes for a given
    device.

    Arguments:
        device (torch.device or int, optional): selected device. Returns
            statistic for the current device, given by :func:`~torch.cuda.current_device`,
            if :attr:`device` is ``None`` (default).

    .. note::
        This is likely less than the amount shown in `nvidia-smi` since some
        unused memory can be held by the caching allocator and some context
        needs to be created on GPU. See :ref:`cuda-memory-management` for more
        details about GPU memory management.
    "
  [ device ]
  (py/call-attr cuda "memory_allocated"  device ))

(defn memory-cached 
  "Returns the current GPU memory managed by the caching allocator in bytes
    for a given device.

    Arguments:
        device (torch.device or int, optional): selected device. Returns
            statistic for the current device, given by :func:`~torch.cuda.current_device`,
            if :attr:`device` is ``None`` (default).

    .. note::
        See :ref:`cuda-memory-management` for more details about GPU memory
        management.
    "
  [ device ]
  (py/call-attr cuda "memory_cached"  device ))

(defn raise-from 
  ""
  [ value from_value ]
  (py/call-attr cuda "raise_from"  value from_value ))

(defn reset-max-memory-allocated 
  "Resets the starting point in tracking maximum GPU memory occupied by
    tensors for a given device.

    See :func:`~torch.cuda.max_memory_allocated` for details.

    Arguments:
        device (torch.device or int, optional): selected device. Returns
            statistic for the current device, given by :func:`~torch.cuda.current_device`,
            if :attr:`device` is ``None`` (default).

    .. note::
        See :ref:`cuda-memory-management` for more details about GPU memory
        management.
    "
  [ device ]
  (py/call-attr cuda "reset_max_memory_allocated"  device ))

(defn reset-max-memory-cached 
  "Resets the starting point in tracking maximum GPU memory managed by the
    caching allocator for a given device.

    See :func:`~torch.cuda.max_memory_cached` for details.

    Arguments:
        device (torch.device or int, optional): selected device. Returns
            statistic for the current device, given by :func:`~torch.cuda.current_device`,
            if :attr:`device` is ``None`` (default).

    .. note::
        See :ref:`cuda-memory-management` for more details about GPU memory
        management.
    "
  [ device ]
  (py/call-attr cuda "reset_max_memory_cached"  device ))

(defn seed 
  "Sets the seed for generating random numbers to a random number for the current GPU.
    It's safe to call this function if CUDA is not available; in that
    case, it is silently ignored.

    .. warning::
        If you are working with a multi-GPU model, this function will only initialize
        the seed on one GPU.  To initialize all GPUs, use :func:`seed_all`.
    "
  [  ]
  (py/call-attr cuda "seed"  ))

(defn seed-all 
  "Sets the seed for generating random numbers to a random number on all GPUs.
    It's safe to call this function if CUDA is not available; in that
    case, it is silently ignored.
    "
  [  ]
  (py/call-attr cuda "seed_all"  ))

(defn set-device 
  "Sets the current device.

    Usage of this function is discouraged in favor of :any:`device`. In most
    cases it's better to use ``CUDA_VISIBLE_DEVICES`` environmental variable.

    Arguments:
        device (torch.device or int): selected device. This function is a no-op
            if this argument is negative.
    "
  [ device ]
  (py/call-attr cuda "set_device"  device ))

(defn set-rng-state 
  "Sets the random number generator state of the specified GPU.

    Args:
        new_state (torch.ByteTensor): The desired state
        device (torch.device or int, optional): The device to set the RNG state.
            Default: ``'cuda'`` (i.e., ``torch.device('cuda')``, the current CUDA device).
    "
  [new_state & {:keys [device]
                       :or {device "cuda"}} ]
    (py/call-attr-kw cuda "set_rng_state" [new_state] {:device device }))

(defn set-rng-state-all 
  "Sets the random number generator state of all devices.

    Args:
        new_state (tuple of torch.ByteTensor): The desired state for each device"
  [ new_states ]
  (py/call-attr cuda "set_rng_state_all"  new_states ))

(defn stream 
  "Context-manager that selects a given stream.

    All CUDA kernels queued within its context will be enqueued on a selected
    stream.

    Arguments:
        stream (Stream): selected stream. This manager is a no-op if it's
            ``None``.

    .. note:: Streams are per-device. If the selected stream is not on the
        current device, this function will also change the current device to
        match the stream.
    "
  [ stream ]
  (py/call-attr cuda "stream"  stream ))

(defn synchronize 
  "Waits for all kernels in all streams on a CUDA device to complete.

    Arguments:
        device (torch.device or int, optional): device for which to synchronize.
            It uses the current device, given by :func:`~torch.cuda.current_device`,
            if :attr:`device` is ``None`` (default).
    "
  [ device ]
  (py/call-attr cuda "synchronize"  device ))
