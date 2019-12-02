(ns torch.distributions.one-hot-categorical.OneHotCategorical
  "
    Creates a one-hot categorical distribution parameterized by :attr:`probs` or
    :attr:`logits`.

    Samples are one-hot coded vectors of size ``probs.size(-1)``.

    .. note:: :attr:`probs` must be non-negative, finite and have a non-zero sum,
              and it will be normalized to sum to 1.

    See also: :func:`torch.distributions.Categorical` for specifications of
    :attr:`probs` and :attr:`logits`.

    Example::

        >>> m = OneHotCategorical(torch.tensor([ 0.25, 0.25, 0.25, 0.25 ]))
        >>> m.sample()  # equal probability of 0, 1, 2, 3
        tensor([ 0.,  0.,  0.,  1.])

    Args:
        probs (Tensor): event probabilities
        logits (Tensor): event log probabilities
    "
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce one-hot-categorical (import-module "torch.distributions.one_hot_categorical"))

(defn OneHotCategorical 
  "
    Creates a one-hot categorical distribution parameterized by :attr:`probs` or
    :attr:`logits`.

    Samples are one-hot coded vectors of size ``probs.size(-1)``.

    .. note:: :attr:`probs` must be non-negative, finite and have a non-zero sum,
              and it will be normalized to sum to 1.

    See also: :func:`torch.distributions.Categorical` for specifications of
    :attr:`probs` and :attr:`logits`.

    Example::

        >>> m = OneHotCategorical(torch.tensor([ 0.25, 0.25, 0.25, 0.25 ]))
        >>> m.sample()  # equal probability of 0, 1, 2, 3
        tensor([ 0.,  0.,  0.,  1.])

    Args:
        probs (Tensor): event probabilities
        logits (Tensor): event log probabilities
    "
  [ probs logits validate_args ]
  (py/call-attr one-hot-categorical "OneHotCategorical"  probs logits validate_args ))

(defn batch-shape 
  "
        Returns the shape over which parameters are batched.
        "
  [ self ]
    (py/call-attr self "batch_shape"))

(defn cdf 
  "
        Returns the cumulative density/mass function evaluated at
        `value`.

        Args:
            value (Tensor):
        "
  [ self value ]
  (py/call-attr self "cdf"  self value ))

(defn entropy 
  ""
  [ self  ]
  (py/call-attr self "entropy"  self  ))

(defn enumerate-support 
  ""
  [self  & {:keys [expand]
                       :or {expand true}} ]
    (py/call-attr-kw self "enumerate_support" [] {:expand expand }))

(defn event-shape 
  "
        Returns the shape of a single sample (without batching).
        "
  [ self ]
    (py/call-attr self "event_shape"))

(defn expand 
  ""
  [ self batch_shape _instance ]
  (py/call-attr self "expand"  self batch_shape _instance ))

(defn icdf 
  "
        Returns the inverse cumulative density/mass function evaluated at
        `value`.

        Args:
            value (Tensor):
        "
  [ self value ]
  (py/call-attr self "icdf"  self value ))

(defn log-prob 
  ""
  [ self value ]
  (py/call-attr self "log_prob"  self value ))

(defn logits 
  ""
  [ self ]
    (py/call-attr self "logits"))

(defn mean 
  ""
  [ self ]
    (py/call-attr self "mean"))

(defn param-shape 
  ""
  [ self ]
    (py/call-attr self "param_shape"))

(defn perplexity 
  "
        Returns perplexity of distribution, batched over batch_shape.

        Returns:
            Tensor of shape batch_shape.
        "
  [ self  ]
  (py/call-attr self "perplexity"  self  ))

(defn probs 
  ""
  [ self ]
    (py/call-attr self "probs"))

(defn rsample 
  "
        Generates a sample_shape shaped reparameterized sample or sample_shape
        shaped batch of reparameterized samples if the distribution parameters
        are batched.
        "
  [self  & {:keys [sample_shape]
                       :or {sample_shape torch.Size([])}} ]
    (py/call-attr-kw self "rsample" [] {:sample_shape sample_shape }))

(defn sample 
  ""
  [self  & {:keys [sample_shape]
                       :or {sample_shape torch.Size([])}} ]
    (py/call-attr-kw self "sample" [] {:sample_shape sample_shape }))

(defn sample-n 
  "
        Generates n samples or n batches of samples if the distribution
        parameters are batched.
        "
  [ self n ]
  (py/call-attr self "sample_n"  self n ))

(defn set-default-validate-args 
  ""
  [ self value ]
  (py/call-attr self "set_default_validate_args"  self value ))

(defn stddev 
  "
        Returns the standard deviation of the distribution.
        "
  [ self ]
    (py/call-attr self "stddev"))

(defn variance 
  ""
  [ self ]
    (py/call-attr self "variance"))
