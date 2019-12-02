(ns torch.autograd
  "
``torch.autograd`` provides classes and functions implementing automatic
differentiation of arbitrary scalar valued functions. It requires minimal
changes to the existing code - you only need to declare :class:`Tensor` s
for which gradients should be computed with the ``requires_grad=True`` keyword.
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

(defn backward 
  "Computes the sum of gradients of given tensors w.r.t. graph leaves.

    The graph is differentiated using the chain rule. If any of ``tensors``
    are non-scalar (i.e. their data has more than one element) and require
    gradient, then the Jacobian-vector product would be computed, in this
    case the function additionally requires specifying ``grad_tensors``.
    It should be a sequence of matching length, that contains the \"vector\"
    in the Jacobian-vector product, usually the gradient of the differentiated
    function w.r.t. corresponding tensors (``None`` is an acceptable value for
    all tensors that don't need gradient tensors).

    This function accumulates gradients in the leaves - you might need to zero
    them before calling it.

    Arguments:
        tensors (sequence of Tensor): Tensors of which the derivative will be
            computed.
        grad_tensors (sequence of (Tensor or None)): The \"vector\" in the Jacobian-vector
            product, usually gradients w.r.t. each element of corresponding tensors.
            None values can be specified for scalar Tensors or ones that don't require
            grad. If a None value would be acceptable for all grad_tensors, then this
            argument is optional.
        retain_graph (bool, optional): If ``False``, the graph used to compute the grad
            will be freed. Note that in nearly all cases setting this option to ``True``
            is not needed and often can be worked around in a much more efficient
            way. Defaults to the value of ``create_graph``.
        create_graph (bool, optional): If ``True``, graph of the derivative will
            be constructed, allowing to compute higher order derivative products.
            Defaults to ``False``.
    "
  [tensors grad_tensors retain_graph & {:keys [create_graph grad_variables]
                       :or {create_graph false}} ]
    (py/call-attr-kw autograd "backward" [tensors grad_tensors retain_graph] {:create_graph create_graph :grad_variables grad_variables }))

(defn grad 
  "Computes and returns the sum of gradients of outputs w.r.t. the inputs.

    ``grad_outputs`` should be a sequence of length matching ``output``
    containing the \"vector\" in Jacobian-vector product, usually the pre-computed
    gradients w.r.t. each of the outputs. If an output doesn't require_grad,
    then the gradient can be ``None``).

    If ``only_inputs`` is ``True``, the function will only return a list of gradients
    w.r.t the specified inputs. If it's ``False``, then gradient w.r.t. all remaining
    leaves will still be computed, and will be accumulated into their ``.grad``
    attribute.

    Arguments:
        outputs (sequence of Tensor): outputs of the differentiated function.
        inputs (sequence of Tensor): Inputs w.r.t. which the gradient will be
            returned (and not accumulated into ``.grad``).
        grad_outputs (sequence of Tensor): The \"vector\" in the Jacobian-vector product.
            Usually gradients w.r.t. each output. None values can be specified for scalar
            Tensors or ones that don't require grad. If a None value would be acceptable
            for all grad_tensors, then this argument is optional. Default: None.
        retain_graph (bool, optional): If ``False``, the graph used to compute the grad
            will be freed. Note that in nearly all cases setting this option to ``True``
            is not needed and often can be worked around in a much more efficient
            way. Defaults to the value of ``create_graph``.
        create_graph (bool, optional): If ``True``, graph of the derivative will
            be constructed, allowing to compute higher order derivative products.
            Default: ``False``.
        allow_unused (bool, optional): If ``False``, specifying inputs that were not
            used when computing outputs (and therefore their grad is always zero)
            is an error. Defaults to ``False``.
    "
  [outputs inputs grad_outputs retain_graph & {:keys [create_graph only_inputs allow_unused]
                       :or {create_graph false only_inputs true allow_unused false}} ]
    (py/call-attr-kw autograd "grad" [outputs inputs grad_outputs retain_graph] {:create_graph create_graph :only_inputs only_inputs :allow_unused allow_unused }))

(defn gradcheck 
  "Check gradients computed via small finite differences against analytical
    gradients w.r.t. tensors in :attr:`inputs` that are of floating point type
    and with ``requires_grad=True``.

    The check between numerical and analytical gradients uses :func:`~torch.allclose`.

    .. note::
        The default values are designed for :attr:`input` of double precision.
        This check will likely fail if :attr:`input` is of less precision, e.g.,
        ``FloatTensor``.

    .. warning::
       If any checked tensor in :attr:`input` has overlapping memory, i.e.,
       different indices pointing to the same memory address (e.g., from
       :func:`torch.expand`), this check will likely fail because the numerical
       gradients computed by point perturbation at such indices will change
       values at all other indices that share the same memory address.

    Args:
        func (function): a Python function that takes Tensor inputs and returns
            a Tensor or a tuple of Tensors
        inputs (tuple of Tensor or Tensor): inputs to the function
        eps (float, optional): perturbation for finite differences
        atol (float, optional): absolute tolerance
        rtol (float, optional): relative tolerance
        raise_exception (bool, optional): indicating whether to raise an exception if
            the check fails. The exception gives more information about the
            exact nature of the failure. This is helpful when debugging gradchecks.
        check_sparse_nnz (bool, optional): if True, gradcheck allows for SparseTensor input,
            and for any SparseTensor at input, gradcheck will perform check at nnz positions only.
        nondet_tol (float, optional): tolerance for non-determinism. When running
            identical inputs through the differentiation, the results must either match
            exactly (default, 0.0) or be within this tolerance.

    Returns:
        True if all differences satisfy allclose condition
    "
  [func inputs & {:keys [eps atol rtol raise_exception check_sparse_nnz nondet_tol]
                       :or {eps 1e-06 atol 1e-05 rtol 0.001 raise_exception true check_sparse_nnz false nondet_tol 0.0}} ]
    (py/call-attr-kw autograd "gradcheck" [func inputs] {:eps eps :atol atol :rtol rtol :raise_exception raise_exception :check_sparse_nnz check_sparse_nnz :nondet_tol nondet_tol }))

(defn gradgradcheck 
  "Check gradients of gradients computed via small finite differences
    against analytical gradients w.r.t. tensors in :attr:`inputs` and
    :attr:`grad_outputs` that are of floating point type and with
    ``requires_grad=True``.

    This function checks that backpropagating through the gradients computed
    to the given :attr:`grad_outputs` are correct.

    The check between numerical and analytical gradients uses :func:`~torch.allclose`.

    .. note::
        The default values are designed for :attr:`input` and
        :attr:`grad_outputs` of double precision. This check will likely fail if
        they are of less precision, e.g., ``FloatTensor``.

    .. warning::
       If any checked tensor in :attr:`input` and :attr:`grad_outputs` has
       overlapping memory, i.e., different indices pointing to the same memory
       address (e.g., from :func:`torch.expand`), this check will likely fail
       because the numerical gradients computed by point perturbation at such
       indices will change values at all other indices that share the same
       memory address.

    Args:
        func (function): a Python function that takes Tensor inputs and returns
            a Tensor or a tuple of Tensors
        inputs (tuple of Tensor or Tensor): inputs to the function
        grad_outputs (tuple of Tensor or Tensor, optional): The gradients with
            respect to the function's outputs.
        eps (float, optional): perturbation for finite differences
        atol (float, optional): absolute tolerance
        rtol (float, optional): relative tolerance
        gen_non_contig_grad_outputs (bool, optional): if :attr:`grad_outputs` is
            ``None`` and :attr:`gen_non_contig_grad_outputs` is ``True``, the
            randomly generated gradient outputs are made to be noncontiguous
        raise_exception (bool, optional): indicating whether to raise an exception if
            the check fails. The exception gives more information about the
            exact nature of the failure. This is helpful when debugging gradchecks.
        nondet_tol (float, optional): tolerance for non-determinism. When running
            identical inputs through the differentiation, the results must either match
            exactly (default, 0.0) or be within this tolerance. Note that a small amount
            of nondeterminism in the gradient will lead to larger inaccuracies in
            the second derivative.

    Returns:
        True if all differences satisfy allclose condition
    "
  [func inputs grad_outputs & {:keys [eps atol rtol gen_non_contig_grad_outputs raise_exception nondet_tol]
                       :or {eps 1e-06 atol 1e-05 rtol 0.001 gen_non_contig_grad_outputs false raise_exception true nondet_tol 0.0}} ]
    (py/call-attr-kw autograd "gradgradcheck" [func inputs grad_outputs] {:eps eps :atol atol :rtol rtol :gen_non_contig_grad_outputs gen_non_contig_grad_outputs :raise_exception raise_exception :nondet_tol nondet_tol }))

(defn variable 
  ""
  [  ]
  (py/call-attr autograd "variable"  ))
