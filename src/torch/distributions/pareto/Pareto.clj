(ns torch.distributions.pareto.Pareto
  "
    Samples from a Pareto Type 1 distribution.

    Example::

        >>> m = Pareto(torch.tensor([1.0]), torch.tensor([1.0]))
        >>> m.sample()  # sample from a Pareto distribution with scale=1 and alpha=1
        tensor([ 1.5623])

    Args:
        scale (float or Tensor): Scale parameter of the distribution
        alpha (float or Tensor): Shape parameter of the distribution
    "
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce pareto (import-module "torch.distributions.pareto"))

(defn Pareto 
  "
    Samples from a Pareto Type 1 distribution.

    Example::

        >>> m = Pareto(torch.tensor([1.0]), torch.tensor([1.0]))
        >>> m.sample()  # sample from a Pareto distribution with scale=1 and alpha=1
        tensor([ 1.5623])

    Args:
        scale (float or Tensor): Scale parameter of the distribution
        alpha (float or Tensor): Shape parameter of the distribution
    "
  [ scale alpha validate_args ]
  (py/call-attr pareto "Pareto"  scale alpha validate_args ))

(defn batch-shape 
  "
        Returns the shape over which parameters are batched.
        "
  [ self ]
    (py/call-attr self "batch_shape"))

(defn cdf 
  "
        Computes the cumulative distribution function by inverting the
        transform(s) and computing the score of the base distribution.
        "
  [ self value ]
  (py/call-attr self "cdf"  self value ))

(defn entropy 
  ""
  [ self  ]
  (py/call-attr self "entropy"  self  ))

(defn enumerate-support 
  "
        Returns tensor containing all values supported by a discrete
        distribution. The result will enumerate over dimension 0, so the shape
        of the result will be `(cardinality,) + batch_shape + event_shape`
        (where `event_shape = ()` for univariate distributions).

        Note that this enumerates over all batched tensors in lock-step
        `[[0, 0], [1, 1], ...]`. With `expand=False`, enumeration happens
        along dim 0, but with the remaining batch dimensions being
        singleton dimensions, `[[0], [1], ..`.

        To iterate over the full Cartesian product use
        `itertools.product(m.enumerate_support())`.

        Args:
            expand (bool): whether to expand the support over the
                batch dims to match the distribution's `batch_shape`.

        Returns:
            Tensor iterating over dimension 0.
        "
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

(defn has-rsample 
  ""
  [ self ]
    (py/call-attr self "has_rsample"))

(defn icdf 
  "
        Computes the inverse cumulative distribution function using
        transform(s) and computing the score of the base distribution.
        "
  [ self value ]
  (py/call-attr self "icdf"  self value ))

(defn log-prob 
  "
        Scores the sample by inverting the transform(s) and computing the score
        using the score of the base distribution and the log abs det jacobian.
        "
  [ self value ]
  (py/call-attr self "log_prob"  self value ))

(defn mean 
  ""
  [ self ]
    (py/call-attr self "mean"))

(defn perplexity 
  "
        Returns perplexity of distribution, batched over batch_shape.

        Returns:
            Tensor of shape batch_shape.
        "
  [ self  ]
  (py/call-attr self "perplexity"  self  ))

(defn rsample 
  "
        Generates a sample_shape shaped reparameterized sample or sample_shape
        shaped batch of reparameterized samples if the distribution parameters
        are batched. Samples first from base distribution and applies
        `transform()` for every transform in the list.
        "
  [self  & {:keys [sample_shape]
                       :or {sample_shape torch.Size([])}} ]
    (py/call-attr-kw self "rsample" [] {:sample_shape sample_shape }))

(defn sample 
  "
        Generates a sample_shape shaped sample or sample_shape shaped batch of
        samples if the distribution parameters are batched. Samples first from
        base distribution and applies `transform()` for every transform in the
        list.
        "
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

(defn support 
  ""
  [ self ]
    (py/call-attr self "support"))

(defn variance 
  ""
  [ self ]
    (py/call-attr self "variance"))
