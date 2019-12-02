(ns torch.jit.annotations
  ""
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce annotations (import-module "torch.jit.annotations"))

(defn ann-to-type 
  ""
  [ ann resolver ]
  (py/call-attr annotations "ann_to_type"  ann resolver ))

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
  (py/call-attr annotations "dedent"  text ))

(defn get-num-params 
  ""
  [ fn loc ]
  (py/call-attr annotations "get_num_params"  fn loc ))

(defn get-signature 
  ""
  [ fn rcb loc ]
  (py/call-attr annotations "get_signature"  fn rcb loc ))

(defn get-source-lines-and-file 
  "
    Wrapper around inspect.getsourcelines and inspect.getsourcefile.

    Returns: (sourcelines, file_lino, filename)
    "
  [ obj ]
  (py/call-attr annotations "get_source_lines_and_file"  obj ))

(defn get-type-line 
  "Tries to find the line containing a comment with the type annotation."
  [ source ]
  (py/call-attr annotations "get_type_line"  source ))

(defn is-dict 
  ""
  [ ann ]
  (py/call-attr annotations "is_dict"  ann ))

(defn is-list 
  ""
  [ ann ]
  (py/call-attr annotations "is_list"  ann ))

(defn is-optional 
  ""
  [ ann ]
  (py/call-attr annotations "is_optional"  ann ))

(defn is-tuple 
  ""
  [ ann ]
  (py/call-attr annotations "is_tuple"  ann ))

(defn parse-type-line 
  "Parses a type annotation specified as a comment.

    Example inputs:
        # type: (Tensor, torch.Tensor) -> Tuple[Tensor]
        # type: (Tensor, Tuple[Tensor, Tensor]) -> Tensor
    "
  [ type_line rcb loc ]
  (py/call-attr annotations "parse_type_line"  type_line rcb loc ))

(defn split-type-line 
  "Splits the comment with the type annotation into parts for argument and return types.

    For example, for an input of:
        # type: (Tensor, torch.Tensor) -> Tuple[Tensor, Tensor]

    This function will return:
        (\"(Tensor, torch.Tensor)\", \"Tuple[Tensor, Tensor]\")

    "
  [ type_line ]
  (py/call-attr annotations "split_type_line"  type_line ))

(defn try-real-annotations 
  "Tries to use the Py3.5+ annotation syntax to get the type."
  [ fn ]
  (py/call-attr annotations "try_real_annotations"  fn ))
