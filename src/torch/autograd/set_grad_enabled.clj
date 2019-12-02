(ns torch.autograd.set-grad-enabled
  "Context-manager that sets gradient calculation to on or off.

    ``set_grad_enabled`` will enable or disable grads based on its argument :attr:`mode`.
    It can be used as a context-manager or as a function.

    When using :class:`~enable_grad` context manager, :class:`~set_grad_enabled(False)`
    has no effect.

    This context manager is thread local; it will not affect computation
    in other threads.

    Arguments:
        mode (bool): Flag whether to enable grad (``True``), or disable
                     (``False``). This can be used to conditionally enable
                     gradients.


    Example::

        >>> x = torch.tensor([1], requires_grad=True)
        >>> is_train = False
        >>> with torch.set_grad_enabled(is_train):
        ...   y = x * 2
        >>> y.requires_grad
        False
        >>> torch.set_grad_enabled(True)
        >>> y = x * 2
        >>> y.requires_grad
        True
        >>> torch.set_grad_enabled(False)
        >>> y = x * 2
        >>> y.requires_grad
        False

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

(defn set-grad-enabled 
  "Context-manager that sets gradient calculation to on or off.

    ``set_grad_enabled`` will enable or disable grads based on its argument :attr:`mode`.
    It can be used as a context-manager or as a function.

    When using :class:`~enable_grad` context manager, :class:`~set_grad_enabled(False)`
    has no effect.

    This context manager is thread local; it will not affect computation
    in other threads.

    Arguments:
        mode (bool): Flag whether to enable grad (``True``), or disable
                     (``False``). This can be used to conditionally enable
                     gradients.


    Example::

        >>> x = torch.tensor([1], requires_grad=True)
        >>> is_train = False
        >>> with torch.set_grad_enabled(is_train):
        ...   y = x * 2
        >>> y.requires_grad
        False
        >>> torch.set_grad_enabled(True)
        >>> y = x * 2
        >>> y.requires_grad
        True
        >>> torch.set_grad_enabled(False)
        >>> y = x * 2
        >>> y.requires_grad
        False

    "
  [ mode ]
  (py/call-attr autograd "set_grad_enabled"  mode ))
