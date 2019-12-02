(ns torch.distributions.constraints.stack
  "
    Constraint functor that applies a sequence of constraints
    `cseq` at the submatrices at dimension `dim`,
    in a way compatible with :func:`torch.stack`.
    "
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce constraints (import-module "torch.distributions.constraints"))

(defn stack 
  "
    Constraint functor that applies a sequence of constraints
    `cseq` at the submatrices at dimension `dim`,
    in a way compatible with :func:`torch.stack`.
    "
  [cseq & {:keys [dim]
                       :or {dim 0}} ]
    (py/call-attr-kw constraints "stack" [cseq] {:dim dim }))

(defn check 
  ""
  [ self value ]
  (py/call-attr self "check"  self value ))
