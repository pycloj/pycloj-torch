(ns torch.distributions.fishersnedecor.Gamma
  "
    Creates a Gamma distribution parameterized by shape :attr:`concentration` and :attr:`rate`.

    Example::

        >>> m = Gamma(torch.tensor([1.0]), torch.tensor([1.0]))
        >>> m.sample()  # Gamma distributed with concentration=1 and rate=1
        tensor([ 0.1046])

    Args:
        concentration (float or Tensor): shape parameter of the distribution
            (often referred to as alpha)
        rate (float or Tensor): rate = 1 / scale of the distribution
            (often referred to as beta)
    "
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce fishersnedecor (import-module "torch.distributions.fishersnedecor"))

(defn Gamma 
  "
    Creates a Gamma distribution parameterized by shape :attr:`concentration` and :attr:`rate`.

    Example::

        >>> m = Gamma(torch.tensor([1.0]), torch.tensor([1.0]))
        >>> m.sample()  # Gamma distributed with concentration=1 and rate=1
        tensor([ 0.1046])

    Args:
        concentration (float or Tensor): shape parameter of the distribution
            (often referred to as alpha)
        rate (float or Tensor): rate = 1 / scale of the distribution
            (often referred to as beta)
    "
  [ concentration rate validate_args ]
  (py/call-attr fishersnedecor "Gamma"  concentration rate validate_args ))

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
  ""
  [self  & {:keys [sample_shape]
                       :or {sample_shape torch.Size([])}} ]
    (py/call-attr-kw self "rsample" [] {:sample_shape sample_shape }))

(defn sample 
  "
        Generates a sample_shape shaped sample or sample_shape shaped batch of
        samples if the distribution parameters are batched.
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

(defn variance 
  ""
  [ self ]
    (py/call-attr self "variance"))
