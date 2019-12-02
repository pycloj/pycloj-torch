(ns torch.jit.frontend
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

(defn build-class-def 
  ""
  [ ctx py_def methods self_name ]
  (py/call-attr frontend "build_class_def"  ctx py_def methods self_name ))

(defn build-def 
  ""
  [ ctx py_def type_line self_name ]
  (py/call-attr frontend "build_def"  ctx py_def type_line self_name ))

(defn build-param 
  ""
  [ ctx py_arg self_name kwarg_only ]
  (py/call-attr frontend "build_param"  ctx py_arg self_name kwarg_only ))

(defn build-param-list 
  ""
  [ ctx py_args self_name ]
  (py/call-attr frontend "build_param_list"  ctx py_args self_name ))

(defn build-stmts 
  ""
  [ ctx stmts ]
  (py/call-attr frontend "build_stmts"  ctx stmts ))

(defn dedent 
  "Remove any common leading whitespace from every line in `text`.

    This can be used to make triple-quoted strings line up with the left
    edge of the display, while still presenting them in the source code
    in indented form.

    Note that tabs and spaces are both treated as whitespace, but they
    are not equal: the lines \"  hello\" and \"\thello\" are
    considered to have no common leading whitespace.

    Entirely blank lines are normalized to a newline character.
    "
  [ text ]
  (py/call-attr frontend "dedent"  text ))

(defn find-before 
  ""
  [ctx pos substr & {:keys [offsets]
                       :or {offsets (0, 0)}} ]
    (py/call-attr-kw frontend "find_before" [ctx pos substr] {:offsets offsets }))

(defn get-default-args 
  ""
  [ fn ]
  (py/call-attr frontend "get_default_args"  fn ))

(defn get-jit-class-def 
  ""
  [ cls self_name ]
  (py/call-attr frontend "get_jit_class_def"  cls self_name ))

(defn get-jit-def 
  ""
  [ fn self_name ]
  (py/call-attr frontend "get_jit_def"  fn self_name ))

(defn get-source-lines-and-file 
  "
    Wrapper around inspect.getsourcelines and inspect.getsourcefile.

    Returns: (sourcelines, file_lino, filename)
    "
  [ obj ]
  (py/call-attr frontend "get_source_lines_and_file"  obj ))

(defn is-reserved-name 
  ""
  [ name ]
  (py/call-attr frontend "is_reserved_name"  name ))
