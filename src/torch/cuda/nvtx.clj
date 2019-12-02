(ns torch.cuda.nvtx
  ""
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce nvtx (import-module "torch.cuda.nvtx"))

(defn mark 
  "
    Describe an instantaneous event that occurred at some point.

    Arguments:
        msg (string): ASCII message to associate with the event.
    "
  [ msg ]
  (py/call-attr nvtx "mark"  msg ))

(defn range-pop 
  "
    Pops a range off of a stack of nested range spans.  Returns the
    zero-based depth of the range that is ended.
    "
  [  ]
  (py/call-attr nvtx "range_pop"  ))

(defn range-push 
  "
    Pushes a range onto a stack of nested range span.  Returns zero-based
    depth of the range that is started.

    Arguments:
        msg (string): ASCII message to associate with range
    "
  [ msg ]
  (py/call-attr nvtx "range_push"  msg ))

(defn windows-nvToolsExt-lib 
  ""
  [  ]
  (py/call-attr nvtx "windows_nvToolsExt_lib"  ))

(defn windows-nvToolsExt-path 
  ""
  [  ]
  (py/call-attr nvtx "windows_nvToolsExt_path"  ))
