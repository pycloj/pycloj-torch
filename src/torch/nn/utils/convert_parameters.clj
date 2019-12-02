(ns torch.nn.utils.convert-parameters
  ""
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce convert-parameters (import-module "torch.nn.utils.convert_parameters"))

(defn parameters-to-vector 
  "Convert parameters to one vector

    Arguments:
        parameters (Iterable[Tensor]): an iterator of Tensors that are the
            parameters of a model.

    Returns:
        The parameters represented by a single vector
    "
  [ parameters ]
  (py/call-attr convert-parameters "parameters_to_vector"  parameters ))

(defn vector-to-parameters 
  "Convert one vector to the parameters

    Arguments:
        vec (Tensor): a single vector represents the parameters of a model.
        parameters (Iterable[Tensor]): an iterator of Tensors that are the
            parameters of a model.
    "
  [ vec parameters ]
  (py/call-attr convert-parameters "vector_to_parameters"  vec parameters ))
