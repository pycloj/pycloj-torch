(ns torch.autograd.set-detect-anomaly
  "Context-manager that sets the anomaly detection for the autograd engine on or off.

    ``set_detect_anomaly`` will enable or disable the autograd anomaly detection
    based on its argument :attr:`mode`.
    It can be used as a context-manager or as a function.

    See ``detect_anomaly`` above for details of the anomaly detection behaviour.

    Arguments:
        mode (bool): Flag whether to enable anomaly detection (``True``),
                     or disable (``False``).

    "
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce autograd (import-module "torch.autograd"))

(defn set-detect-anomaly 
  "Context-manager that sets the anomaly detection for the autograd engine on or off.

    ``set_detect_anomaly`` will enable or disable the autograd anomaly detection
    based on its argument :attr:`mode`.
    It can be used as a context-manager or as a function.

    See ``detect_anomaly`` above for details of the anomaly detection behaviour.

    Arguments:
        mode (bool): Flag whether to enable anomaly detection (``True``),
                     or disable (``False``).

    "
  [ mode ]
  (py/call-attr autograd "set_detect_anomaly"  mode ))
