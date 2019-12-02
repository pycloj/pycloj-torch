(ns torch.distributions.relaxed-bernoulli.SigmoidTransform
  "
    Transform via the mapping :math:`y = \frac{1}{1 + \exp(-x)}` and :math:`x = \text{logit}(y)`.
    "
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce relaxed-bernoulli (import-module "torch.distributions.relaxed_bernoulli"))

(defn SigmoidTransform 
  "
    Transform via the mapping :math:`y = \frac{1}{1 + \exp(-x)}` and :math:`x = \text{logit}(y)`.
    "
  [ & {:keys [cache_size]
       :or {cache_size 0}} ]
  
   (py/call-attr-kw relaxed-bernoulli "SigmoidTransform" [] {:cache_size cache_size }))

(defn inv 
  "
        Returns the inverse :class:`Transform` of this transform.
        This should satisfy ``t.inv.inv is t``.
        "
  [ self ]
    (py/call-attr self "inv"))

(defn log-abs-det-jacobian 
  ""
  [ self x y ]
  (py/call-attr self "log_abs_det_jacobian"  self x y ))
