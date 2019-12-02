(ns torch.backends
  ""
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce backends (import-module "torch.backends"))

(defn contextmanager 
  "@contextmanager decorator.

    Typical usage:

        @contextmanager
        def some_generator(<arguments>):
            <setup>
            try:
                yield <value>
            finally:
                <cleanup>

    This makes this:

        with some_generator(<arguments>) as <variable>:
            <body>

    equivalent to this:

        <setup>
        try:
            <variable> = <value>
            <body>
        finally:
            <cleanup>
    "
  [ func ]
  (py/call-attr backends "contextmanager"  func ))

(defn disable-global-flags 
  ""
  [  ]
  (py/call-attr backends "disable_global_flags"  ))

(defn flags-frozen 
  ""
  [  ]
  (py/call-attr backends "flags_frozen"  ))
