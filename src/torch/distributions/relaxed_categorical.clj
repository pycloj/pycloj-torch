(ns torch.distributions.relaxed-categorical
  ""
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce relaxed-categorical (import-module "torch.distributions.relaxed_categorical"))

(defn broadcast-all 
  "
    Given a list of values (possibly containing numbers), returns a list where each
    value is broadcasted based on the following rules:
      - `torch.*Tensor` instances are broadcasted as per :ref:`_broadcasting-semantics`.
      - numbers.Number instances (scalars) are upcast to tensors having
        the same size and type as the first tensor passed to `values`.  If all the
        values are scalars, then they are upcasted to scalar Tensors.

    Args:
        values (list of `numbers.Number` or `torch.*Tensor`)

    Raises:
        ValueError: if any of the values is not a `numbers.Number` or
            `torch.*Tensor` instance
    "
  [  ]
  (py/call-attr relaxed-categorical "broadcast_all"  ))

(defn clamp-probs 
  ""
  [ probs ]
  (py/call-attr relaxed-categorical "clamp_probs"  probs ))
