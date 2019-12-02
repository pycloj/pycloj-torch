(ns torch.jit.frontend.ExprBuilder
  ""
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce frontend (import-module "torch.jit.frontend"))

(defn ExprBuilder 
  ""
  [  ]
  (py/call-attr frontend "ExprBuilder"  ))

(defn build-Attribute 
  ""
  [ self ctx expr ]
  (py/call-attr self "build_Attribute"  self ctx expr ))

(defn build-BinOp 
  ""
  [ self ctx expr ]
  (py/call-attr self "build_BinOp"  self ctx expr ))

(defn build-BoolOp 
  ""
  [ self ctx expr ]
  (py/call-attr self "build_BoolOp"  self ctx expr ))

(defn build-Call 
  ""
  [ self ctx expr ]
  (py/call-attr self "build_Call"  self ctx expr ))

(defn build-Compare 
  ""
  [ self ctx expr ]
  (py/call-attr self "build_Compare"  self ctx expr ))

(defn build-Constant 
  ""
  [ self ctx expr ]
  (py/call-attr self "build_Constant"  self ctx expr ))

(defn build-Dict 
  ""
  [ self ctx expr ]
  (py/call-attr self "build_Dict"  self ctx expr ))

(defn build-Ellipsis 
  ""
  [ self ctx expr ]
  (py/call-attr self "build_Ellipsis"  self ctx expr ))

(defn build-IfExp 
  ""
  [ self ctx expr ]
  (py/call-attr self "build_IfExp"  self ctx expr ))

(defn build-JoinedStr 
  ""
  [ self ctx expr ]
  (py/call-attr self "build_JoinedStr"  self ctx expr ))

(defn build-List 
  ""
  [ self ctx expr ]
  (py/call-attr self "build_List"  self ctx expr ))

(defn build-ListComp 
  ""
  [ self ctx stmt ]
  (py/call-attr self "build_ListComp"  self ctx stmt ))

(defn build-Name 
  ""
  [ self ctx expr ]
  (py/call-attr self "build_Name"  self ctx expr ))

(defn build-NameConstant 
  ""
  [ self ctx expr ]
  (py/call-attr self "build_NameConstant"  self ctx expr ))

(defn build-Num 
  ""
  [ self ctx expr ]
  (py/call-attr self "build_Num"  self ctx expr ))

(defn build-Starred 
  ""
  [ self ctx expr ]
  (py/call-attr self "build_Starred"  self ctx expr ))

(defn build-Str 
  ""
  [ self ctx expr ]
  (py/call-attr self "build_Str"  self ctx expr ))

(defn build-Subscript 
  ""
  [ self ctx expr ]
  (py/call-attr self "build_Subscript"  self ctx expr ))

(defn build-Tuple 
  ""
  [ self ctx expr ]
  (py/call-attr self "build_Tuple"  self ctx expr ))

(defn build-UnaryOp 
  ""
  [ self ctx expr ]
  (py/call-attr self "build_UnaryOp"  self ctx expr ))
