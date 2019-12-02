(ns torch.distributed.internal-rpc-utils
  ""
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce internal-rpc-utils (import-module "torch.distributed.internal_rpc_utils"))

(defn load-python-udf-result-internal 
  ""
  [ pickled_python_result ]
  (py/call-attr internal-rpc-utils "load_python_udf_result_internal"  pickled_python_result ))

(defn run-python-udf-internal 
  ""
  [ pickled_python_udf ]
  (py/call-attr internal-rpc-utils "run_python_udf_internal"  pickled_python_udf ))

(defn serialize 
  ""
  [ obj ]
  (py/call-attr internal-rpc-utils "serialize"  obj ))
