(ns torch.cuda.profiler
  ""
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce profiler (import-module "torch.cuda.profiler"))

(defn check-error 
  ""
  [ res ]
  (py/call-attr profiler "check_error"  res ))

(defn cudart 
  ""
  [  ]
  (py/call-attr profiler "cudart"  ))

(defn init 
  ""
  [output_file flags & {:keys [output_mode]
                       :or {output_mode "key_value"}} ]
    (py/call-attr-kw profiler "init" [output_file flags] {:output_mode output_mode }))

(defn profile 
  ""
  [  ]
  (py/call-attr profiler "profile"  ))

(defn start 
  ""
  [  ]
  (py/call-attr profiler "start"  ))

(defn stop 
  ""
  [  ]
  (py/call-attr profiler "stop"  ))
