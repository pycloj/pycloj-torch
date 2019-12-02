(ns torch.jit.CompilationUnit
  ""
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce jit (import-module "torch.jit"))

(defn CompilationUnit 
  ""
  [lang & {:keys [_frames_up]
                       :or {_frames_up 0}} ]
    (py/call-attr-kw jit "CompilationUnit" [lang] {:_frames_up _frames_up }))

(defn define 
  ""
  [self lang rcb & {:keys [_frames_up]
                       :or {_frames_up 0}} ]
    (py/call-attr-kw self "define" [lang rcb] {:_frames_up _frames_up }))
