(ns torch.Gradient
  ""
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce torch (import-module "torch"))

(defn df 
  ""
  [ self ]
    (py/call-attr self "df"))

(defn df-input-captured-inputs 
  ""
  [ self ]
    (py/call-attr self "df_input_captured_inputs"))

(defn df-input-captured-outputs 
  ""
  [ self ]
    (py/call-attr self "df_input_captured_outputs"))

(defn df-input-vjps 
  ""
  [ self ]
    (py/call-attr self "df_input_vjps"))

(defn df-output-vjps 
  ""
  [ self ]
    (py/call-attr self "df_output_vjps"))

(defn f 
  ""
  [ self ]
    (py/call-attr self "f"))

(defn f-real-outputs 
  ""
  [ self ]
    (py/call-attr self "f_real_outputs"))
