(ns torch.backends.cudnn
  ""
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce m (import-module "torch.backends.cudnn"))

(defn add-tensor 
  ""
  [  ]
  (py/call-attr m "add_tensor"  ))

(defn c-type 
  ""
  [ tensor ]
  (py/call-attr m "c_type"  tensor ))

(defn check-error 
  ""
  [ status ]
  (py/call-attr m "check_error"  status ))

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
  (py/call-attr m "contextmanager"  func ))

(defn descriptor 
  ""
  [ tensor N ]
  (py/call-attr m "descriptor"  tensor N ))

(defn descriptor-sequence 
  ""
  [ tensor batch_sizes ]
  (py/call-attr m "descriptor_sequence"  tensor batch_sizes ))

(defn find-cudnn-windows-lib 
  ""
  [  ]
  (py/call-attr m "find_cudnn_windows_lib"  ))

(defn flags 
  ""
  [ & {:keys [enabled benchmark deterministic verbose]
       :or {enabled false benchmark false deterministic false verbose false}} ]
  
   (py/call-attr-kw m "flags" [] {:enabled enabled :benchmark benchmark :deterministic deterministic :verbose verbose }))

(defn get-error-string 
  ""
  [ status ]
  (py/call-attr m "get_error_string"  status ))

(defn get-handle 
  ""
  [  ]
  (py/call-attr m "get_handle"  ))

(defn int-array 
  ""
  [ itr ]
  (py/call-attr m "int_array"  itr ))

(defn is-acceptable 
  ""
  [ tensor ]
  (py/call-attr m "is_acceptable"  tensor ))

(defn is-available 
  "Returns a bool indicating if CUDNN is currently available."
  [  ]
  (py/call-attr m "is_available"  ))

(defn set-flags 
  ""
  [ _enabled _benchmark _deterministic _verbose ]
  (py/call-attr m "set_flags"  _enabled _benchmark _deterministic _verbose ))

(defn version 
  ""
  [  ]
  (py/call-attr m "version"  ))
