(ns torch.Generator
  "
Generator(device='cpu') -> Generator

Creates and returns a generator object which manages the state of the algorithm that
produces pseudo random numbers. Used as a keyword argument in many :ref:`inplace-random-sampling`
functions.

Arguments:
    device (:class:`torch.device`, optional): the desired device for the generator.

Returns:
    Generator: An torch.Generator object.

Example::

    >>> g_cpu = torch.Generator()
    >>> g_cuda = torch.Generator(device='cuda')
"
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce torch (import-module "torch"))
