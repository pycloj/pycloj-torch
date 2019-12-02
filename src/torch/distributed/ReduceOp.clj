(ns torch.distributed.ReduceOp
  "
An enum-like class for available reduction operations: ``SUM``, ``PRODUCT``,
``MIN``, ``MAX``, ``BAND``, ``BOR``, and ``BXOR``.

The values of this class can be accessed as attributes, e.g., ``ReduceOp.SUM``.
They are used in specifying strategies for reduction collectives, e.g.,
:func:`reduce`, :func:`all_reduce_multigpu`, etc.

Members:

  SUM

  PRODUCT

  MIN

  MAX

  BAND

  BOR

  BXOR"
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce distributed (import-module "torch.distributed"))

(defn name 
  "(self: handle) -> str
"
  [ self ]
    (py/call-attr self "name"))
