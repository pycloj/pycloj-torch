(ns torch.multiprocessing.reductions
  ""
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce reductions (import-module "torch.multiprocessing.reductions"))

(defn check-serializing-named-tensor 
  ""
  [ tensor ]
  (py/call-attr reductions "check_serializing_named_tensor"  tensor ))

(defn fd-id 
  ""
  [ fd ]
  (py/call-attr reductions "fd_id"  fd ))

(defn init-reductions 
  ""
  [  ]
  (py/call-attr reductions "init_reductions"  ))

(defn rebuild-cuda-tensor 
  ""
  [ tensor_cls tensor_size tensor_stride tensor_offset storage_cls storage_device storage_handle storage_size_bytes storage_offset_bytes requires_grad ref_counter_handle ref_counter_offset event_handle event_sync_required ]
  (py/call-attr reductions "rebuild_cuda_tensor"  tensor_cls tensor_size tensor_stride tensor_offset storage_cls storage_device storage_handle storage_size_bytes storage_offset_bytes requires_grad ref_counter_handle ref_counter_offset event_handle event_sync_required ))

(defn rebuild-event 
  ""
  [ device handle ]
  (py/call-attr reductions "rebuild_event"  device handle ))

(defn rebuild-storage-empty 
  ""
  [ cls ]
  (py/call-attr reductions "rebuild_storage_empty"  cls ))

(defn rebuild-storage-fd 
  ""
  [ cls df size ]
  (py/call-attr reductions "rebuild_storage_fd"  cls df size ))

(defn rebuild-storage-filename 
  ""
  [ cls manager handle size ]
  (py/call-attr reductions "rebuild_storage_filename"  cls manager handle size ))

(defn rebuild-tensor 
  ""
  [ cls storage metadata ]
  (py/call-attr reductions "rebuild_tensor"  cls storage metadata ))

(defn reduce-event 
  ""
  [ event ]
  (py/call-attr reductions "reduce_event"  event ))

(defn reduce-storage 
  ""
  [ storage ]
  (py/call-attr reductions "reduce_storage"  storage ))

(defn reduce-tensor 
  ""
  [ tensor ]
  (py/call-attr reductions "reduce_tensor"  tensor ))

(defn register-after-fork 
  ""
  [ obj func ]
  (py/call-attr reductions "register_after_fork"  obj func ))

(defn storage-from-cache 
  ""
  [ cls key ]
  (py/call-attr reductions "storage_from_cache"  cls key ))
