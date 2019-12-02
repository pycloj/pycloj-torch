(ns torch.distributed.distributed-c10d.dist-backend
  "
    An enum-like class of available backends: GLOO, NCCL, and MPI.

    The values of this class are lowercase strings, e.g., ``\"gloo\"``. They can
    be accessed as attributes, e.g., ``Backend.NCCL``.

    This class can be directly called to parse the string, e.g.,
    ``Backend(backend_str)`` will check if ``backend_str`` is valid, and
    return the parsed lowercase string if so. It also accepts uppercase strings,
    e.g., ``Backend(\"GLOO\")`` returns ``\"gloo\"``.

    .. note:: The entry ``Backend.UNDEFINED`` is present but only used as
              initial value of some fields. Users should neither use it directly
              nor assume its existence.
    "
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce distributed-c10d (import-module "torch.distributed.distributed_c10d"))

(defn dist-backend 
  "
    An enum-like class of available backends: GLOO, NCCL, and MPI.

    The values of this class are lowercase strings, e.g., ``\"gloo\"``. They can
    be accessed as attributes, e.g., ``Backend.NCCL``.

    This class can be directly called to parse the string, e.g.,
    ``Backend(backend_str)`` will check if ``backend_str`` is valid, and
    return the parsed lowercase string if so. It also accepts uppercase strings,
    e.g., ``Backend(\"GLOO\")`` returns ``\"gloo\"``.

    .. note:: The entry ``Backend.UNDEFINED`` is present but only used as
              initial value of some fields. Users should neither use it directly
              nor assume its existence.
    "
  [ name ]
  (py/call-attr distributed-c10d "dist_backend"  name ))
