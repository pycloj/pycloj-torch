(ns torch.autograd.anomaly-mode.detect-anomaly
  "Context-manager that enable anomaly detection for the autograd engine.

    This does two things:
    - Running the forward pass with detection enabled will allow the backward
    pass to print the traceback of the forward operation that created the failing
    backward function.
    - Any backward computation that generate \"nan\" value will raise an error.

    .. warning::
        This mode should be enabled only for debugging as the different tests
        will slow down your program execution.

    Example:

        >>> import torch
        >>> from torch import autograd
        >>> class MyFunc(autograd.Function):
        ...     @staticmethod
        ...     def forward(ctx, inp):
        ...         return inp.clone()
        ...     @staticmethod
        ...     def backward(ctx, gO):
        ...         # Error during the backward pass
        ...         raise RuntimeError(\"Some error in backward\")
        ...         return gO.clone()
        >>> def run_fn(a):
        ...     out = MyFunc.apply(a)
        ...     return out.sum()
        >>> inp = torch.rand(10, 10, requires_grad=True)
        >>> out = run_fn(inp)
        >>> out.backward()
            Traceback (most recent call last):
              File \"<stdin>\", line 1, in <module>
              File \"/your/pytorch/install/torch/tensor.py\", line 93, in backward
                torch.autograd.backward(self, gradient, retain_graph, create_graph)
              File \"/your/pytorch/install/torch/autograd/__init__.py\", line 90, in backward
                allow_unreachable=True)  # allow_unreachable flag
              File \"/your/pytorch/install/torch/autograd/function.py\", line 76, in apply
                return self._forward_cls.backward(self, *args)
              File \"<stdin>\", line 8, in backward
            RuntimeError: Some error in backward
        >>> with autograd.detect_anomaly():
        ...     inp = torch.rand(10, 10, requires_grad=True)
        ...     out = run_fn(inp)
        ...     out.backward()
            Traceback of forward call that caused the error:
              File \"tmp.py\", line 53, in <module>
                out = run_fn(inp)
              File \"tmp.py\", line 44, in run_fn
                out = MyFunc.apply(a)
            Traceback (most recent call last):
              File \"<stdin>\", line 4, in <module>
              File \"/your/pytorch/install/torch/tensor.py\", line 93, in backward
                torch.autograd.backward(self, gradient, retain_graph, create_graph)
              File \"/your/pytorch/install/torch/autograd/__init__.py\", line 90, in backward
                allow_unreachable=True)  # allow_unreachable flag
              File \"/your/pytorch/install/torch/autograd/function.py\", line 76, in apply
                return self._forward_cls.backward(self, *args)
              File \"<stdin>\", line 8, in backward
            RuntimeError: Some error in backward

    "
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce anomaly-mode (import-module "torch.autograd.anomaly_mode"))

(defn detect-anomaly 
  "Context-manager that enable anomaly detection for the autograd engine.

    This does two things:
    - Running the forward pass with detection enabled will allow the backward
    pass to print the traceback of the forward operation that created the failing
    backward function.
    - Any backward computation that generate \"nan\" value will raise an error.

    .. warning::
        This mode should be enabled only for debugging as the different tests
        will slow down your program execution.

    Example:

        >>> import torch
        >>> from torch import autograd
        >>> class MyFunc(autograd.Function):
        ...     @staticmethod
        ...     def forward(ctx, inp):
        ...         return inp.clone()
        ...     @staticmethod
        ...     def backward(ctx, gO):
        ...         # Error during the backward pass
        ...         raise RuntimeError(\"Some error in backward\")
        ...         return gO.clone()
        >>> def run_fn(a):
        ...     out = MyFunc.apply(a)
        ...     return out.sum()
        >>> inp = torch.rand(10, 10, requires_grad=True)
        >>> out = run_fn(inp)
        >>> out.backward()
            Traceback (most recent call last):
              File \"<stdin>\", line 1, in <module>
              File \"/your/pytorch/install/torch/tensor.py\", line 93, in backward
                torch.autograd.backward(self, gradient, retain_graph, create_graph)
              File \"/your/pytorch/install/torch/autograd/__init__.py\", line 90, in backward
                allow_unreachable=True)  # allow_unreachable flag
              File \"/your/pytorch/install/torch/autograd/function.py\", line 76, in apply
                return self._forward_cls.backward(self, *args)
              File \"<stdin>\", line 8, in backward
            RuntimeError: Some error in backward
        >>> with autograd.detect_anomaly():
        ...     inp = torch.rand(10, 10, requires_grad=True)
        ...     out = run_fn(inp)
        ...     out.backward()
            Traceback of forward call that caused the error:
              File \"tmp.py\", line 53, in <module>
                out = run_fn(inp)
              File \"tmp.py\", line 44, in run_fn
                out = MyFunc.apply(a)
            Traceback (most recent call last):
              File \"<stdin>\", line 4, in <module>
              File \"/your/pytorch/install/torch/tensor.py\", line 93, in backward
                torch.autograd.backward(self, gradient, retain_graph, create_graph)
              File \"/your/pytorch/install/torch/autograd/__init__.py\", line 90, in backward
                allow_unreachable=True)  # allow_unreachable flag
              File \"/your/pytorch/install/torch/autograd/function.py\", line 76, in apply
                return self._forward_cls.backward(self, *args)
              File \"<stdin>\", line 8, in backward
            RuntimeError: Some error in backward

    "
  [  ]
  (py/call-attr anomaly-mode "detect_anomaly"  ))
