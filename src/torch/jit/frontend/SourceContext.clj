(ns torch.jit.frontend.SourceContext
  ""
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce frontend (import-module "torch.jit.frontend"))

(defn SourceContext 
  ""
  [source filename file_lineno leading_whitespace_len & {:keys [uses_true_division]
                       :or {uses_true_division true}} ]
    (py/call-attr-kw frontend "SourceContext" [source filename file_lineno leading_whitespace_len] {:uses_true_division uses_true_division }))

(defn source 
  ""
  [ self ]
    (py/call-attr self "source"))
