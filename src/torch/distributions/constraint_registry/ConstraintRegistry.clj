(ns torch.distributions.constraint-registry.ConstraintRegistry
  "
    Registry to link constraints to transforms.
    "
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce constraint-registry (import-module "torch.distributions.constraint_registry"))

(defn ConstraintRegistry 
  "
    Registry to link constraints to transforms.
    "
  [  ]
  (py/call-attr constraint-registry "ConstraintRegistry"  ))

(defn register 
  "
        Registers a :class:`~torch.distributions.constraints.Constraint`
        subclass in this registry. Usage::

            @my_registry.register(MyConstraintClass)
            def construct_transform(constraint):
                assert isinstance(constraint, MyConstraint)
                return MyTransform(constraint.arg_constraints)

        Args:
            constraint (subclass of :class:`~torch.distributions.constraints.Constraint`):
                A subclass of :class:`~torch.distributions.constraints.Constraint`, or
                a singleton object of the desired class.
            factory (callable): A callable that inputs a constraint object and returns
                a  :class:`~torch.distributions.transforms.Transform` object.
        "
  [ self constraint factory ]
  (py/call-attr self "register"  self constraint factory ))
