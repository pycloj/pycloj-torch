(ns torch.distributions.relaxed-bernoulli
  ""
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce relaxed-bernoulli (import-module "torch.distributions.relaxed_bernoulli"))

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
  (py/call-attr relaxed-bernoulli "broadcast_all"  ))

(defn clamp-probs 
  ""
  [ probs ]
  (py/call-attr relaxed-bernoulli "clamp_probs"  probs ))

(defn logits-to-probs 
  "
    Converts a tensor of logits into probabilities. Note that for the
    binary case, each value denotes log odds, whereas for the
    multi-dimensional case, the values along the last dimension denote
    the log probabilities (possibly unnormalized) of the events.
    "
  [logits & {:keys [is_binary]
                       :or {is_binary false}} ]
    (py/call-attr-kw relaxed-bernoulli "logits_to_probs" [logits] {:is_binary is_binary }))

(defn probs-to-logits 
  "
    Converts a tensor of probabilities into logits. For the binary case,
    this denotes the probability of occurrence of the event indexed by `1`.
    For the multi-dimensional case, the values along the last dimension
    denote the probabilities of occurrence of each of the events.
    "
  [probs & {:keys [is_binary]
                       :or {is_binary false}} ]
    (py/call-attr-kw relaxed-bernoulli "probs_to_logits" [probs] {:is_binary is_binary }))
