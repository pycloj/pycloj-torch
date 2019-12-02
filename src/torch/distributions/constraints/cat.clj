(ns torch.distributions.constraints.cat
  "
    Constraint functor that applies a sequence of constraints
    `cseq` at the submatrices at dimension `dim`,
    each of size `lengths[dim]`, in a way compatible with :func:`torch.cat`.
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

(defn cat 
  "
    Constraint functor that applies a sequence of constraints
    `cseq` at the submatrices at dimension `dim`,
    each of size `lengths[dim]`, in a way compatible with :func:`torch.cat`.
    "
  [cseq & {:keys [dim lengths]
                       :or {dim 0}} ]
    (py/call-attr-kw constraints "cat" [cseq] {:dim dim :lengths lengths }))

(defn check 
  ""
  [ self value ]
  (py/call-attr self "check"  self value ))
