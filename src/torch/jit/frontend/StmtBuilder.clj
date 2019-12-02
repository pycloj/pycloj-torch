(ns torch.jit.frontend.StmtBuilder
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

(defn StmtBuilder 
  ""
  [  ]
  (py/call-attr frontend "StmtBuilder"  ))

(defn build-AnnAssign 
  ""
  [ self ctx stmt ]
  (py/call-attr self "build_AnnAssign"  self ctx stmt ))

(defn build-Assert 
  ""
  [ self ctx stmt ]
  (py/call-attr self "build_Assert"  self ctx stmt ))

(defn build-Assign 
  ""
  [ self ctx stmt ]
  (py/call-attr self "build_Assign"  self ctx stmt ))

(defn build-AugAssign 
  ""
  [ self ctx stmt ]
  (py/call-attr self "build_AugAssign"  self ctx stmt ))

(defn build-Break 
  ""
  [ self ctx stmt ]
  (py/call-attr self "build_Break"  self ctx stmt ))

(defn build-Continue 
  ""
  [ self ctx stmt ]
  (py/call-attr self "build_Continue"  self ctx stmt ))

(defn build-Expr 
  ""
  [ self ctx stmt ]
  (py/call-attr self "build_Expr"  self ctx stmt ))

(defn build-For 
  ""
  [ self ctx stmt ]
  (py/call-attr self "build_For"  self ctx stmt ))

(defn build-If 
  ""
  [ self ctx stmt ]
  (py/call-attr self "build_If"  self ctx stmt ))

(defn build-Pass 
  ""
  [ self ctx stmt ]
  (py/call-attr self "build_Pass"  self ctx stmt ))

(defn build-Print 
  ""
  [ self ctx stmt ]
  (py/call-attr self "build_Print"  self ctx stmt ))

(defn build-Raise 
  ""
  [ self ctx stmt ]
  (py/call-attr self "build_Raise"  self ctx stmt ))

(defn build-Return 
  ""
  [ self ctx stmt ]
  (py/call-attr self "build_Return"  self ctx stmt ))

(defn build-While 
  ""
  [ self ctx stmt ]
  (py/call-attr self "build_While"  self ctx stmt ))
