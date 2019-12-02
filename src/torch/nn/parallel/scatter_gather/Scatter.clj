(ns torch.nn.parallel.scatter-gather.Scatter
  ""
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce scatter-gather (import-module "torch.nn.parallel.scatter_gather"))

(defn backward 
  ""
  [ self ctx ]
  (py/call-attr self "backward"  self ctx ))

(defn forward 
  ""
  [ self ctx target_gpus chunk_sizes dim input ]
  (py/call-attr self "forward"  self ctx target_gpus chunk_sizes dim input ))

(defn mark-dirty 
  "Marks given tensors as modified in an in-place operation.

        **This should be called at most once, only from inside the**
        :func:`forward` **method, and all arguments should be inputs.**

        Every tensor that's been modified in-place in a call to :func:`forward`
        should be given to this function, to ensure correctness of our checks.
        It doesn't matter whether the function is called before or after
        modification.
        "
  [ self  ]
  (py/call-attr self "mark_dirty"  self  ))

(defn mark-non-differentiable 
  "Marks outputs as non-differentiable.

        **This should be called at most once, only from inside the**
        :func:`forward` **method, and all arguments should be outputs.**

        This will mark outputs as not requiring gradients, increasing the
        efficiency of backward computation. You still need to accept a gradient
        for each output in :meth:`~Function.backward`, but it's always going to
        be a zero tensor with the same shape as the shape of a corresponding
        output.

        This is used e.g. for indices returned from a max :class:`Function`.
        "
  [ self  ]
  (py/call-attr self "mark_non_differentiable"  self  ))

(defn mark-shared-storage 
  ""
  [ self  ]
  (py/call-attr self "mark_shared_storage"  self  ))

(defn save-for-backward 
  "Saves given tensors for a future call to :func:`~Function.backward`.

        **This should be called at most once, and only from inside the**
        :func:`forward` **method.**

        Later, saved tensors can be accessed through the :attr:`saved_tensors`
        attribute. Before returning them to the user, a check is made to ensure
        they weren't used in any in-place operation that modified their content.

        Arguments can also be ``None``.
        "
  [ self  ]
  (py/call-attr self "save_for_backward"  self  ))
