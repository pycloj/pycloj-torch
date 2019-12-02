(ns torch.cuda.BFloat16Storage
  ""
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce cuda (import-module "torch.cuda"))

(defn BFloat16Storage 
  ""
  [  ]
  (py/call-attr cuda "BFloat16Storage"  ))

(defn bfloat16 
  "Casts this storage to bfloat16 type"
  [ self  ]
  (py/call-attr self "bfloat16"  self  ))

(defn bool 
  "Casts this storage to bool type"
  [ self  ]
  (py/call-attr self "bool"  self  ))

(defn byte 
  "Casts this storage to byte type"
  [ self  ]
  (py/call-attr self "byte"  self  ))

(defn char 
  "Casts this storage to char type"
  [ self  ]
  (py/call-attr self "char"  self  ))

(defn clone 
  "Returns a copy of this storage"
  [ self  ]
  (py/call-attr self "clone"  self  ))

(defn cpu 
  "Returns a CPU copy of this storage if it's not already on the CPU"
  [ self  ]
  (py/call-attr self "cpu"  self  ))

(defn cuda 
  "Returns a copy of this object in CUDA memory.

    If this object is already in CUDA memory and on the correct device, then
    no copy is performed and the original object is returned.

    Args:
        device (int): The destination GPU id. Defaults to the current device.
        non_blocking (bool): If ``True`` and the source is in pinned memory,
            the copy will be asynchronous with respect to the host. Otherwise,
            the argument has no effect.
        **kwargs: For compatibility, may contain the key ``async`` in place of
            the ``non_blocking`` argument.
    "
  [self device & {:keys [non_blocking]
                       :or {non_blocking false}} ]
    (py/call-attr-kw self "cuda" [device] {:non_blocking non_blocking }))

(defn double 
  "Casts this storage to double type"
  [ self  ]
  (py/call-attr self "double"  self  ))

(defn float 
  "Casts this storage to float type"
  [ self  ]
  (py/call-attr self "float"  self  ))

(defn half 
  "Casts this storage to half type"
  [ self  ]
  (py/call-attr self "half"  self  ))

(defn int 
  "Casts this storage to int type"
  [ self  ]
  (py/call-attr self "int"  self  ))

(defn long 
  "Casts this storage to long type"
  [ self  ]
  (py/call-attr self "long"  self  ))

(defn pin-memory 
  "Copies the storage to pinned memory, if it's not already pinned."
  [ self  ]
  (py/call-attr self "pin_memory"  self  ))

(defn share-memory- 
  "Moves the storage to shared memory.

        This is a no-op for storages already in shared memory and for CUDA
        storages, which do not need to be moved for sharing across processes.
        Storages in shared memory cannot be resized.

        Returns: self
        "
  [ self  ]
  (py/call-attr self "share_memory_"  self  ))

(defn short 
  "Casts this storage to short type"
  [ self  ]
  (py/call-attr self "short"  self  ))

(defn tolist 
  "Returns a list containing the elements of this storage"
  [ self  ]
  (py/call-attr self "tolist"  self  ))

(defn type 
  ""
  [ self  ]
  (py/call-attr self "type"  self  ))
