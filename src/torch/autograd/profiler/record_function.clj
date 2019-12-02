(ns torch.autograd.profiler.record-function
  "Context manager that adds a label to a block of Python code when running autograd
    profiler.  It is useful when tracing the code profile.

    Arguments:
        name (str): Label assigned to the block of code.

    Example:
        >>> x = torch.randn((1, 1), requires_grad=True)
        >>> with torch.autograd.profiler.profile() as prof:
        ...     y = x ** 2
        ...     with torch.autograd.profiler.record_function(\"label-z\"): # label the block
        ...         z = y ** 3
        ...     y.backward()
        ...
        >>> # NOTE: some columns were removed for brevity
        >>> print(prof.key_averages().table(sort_by=\"self_cpu_time_total\"))
        -----------------------------------  ---------------  ---------------  ---------------
        Name                                 Self CPU total %  CPU time avg     Number of Calls
        -----------------------------------  ---------------  ---------------  ---------------
        pow                                  60.77%           47.470us         3
        mul                                  21.73%           25.465us         2
        PowBackward0                         12.03%           121.891us        1
        torch::autograd::AccumulateGrad      2.70%            6.324us          1
        label-z                              2.13%            12.421us         1
        torch::autograd::GraphRoot           0.64%            1.503us          1
        -----------------------------------  ---------------  ---------------  ---------------
        Self CPU time total: 234.344us
        CUDA time total: 0.000us
    "
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce profiler (import-module "torch.autograd.profiler"))

(defn record-function 
  "Context manager that adds a label to a block of Python code when running autograd
    profiler.  It is useful when tracing the code profile.

    Arguments:
        name (str): Label assigned to the block of code.

    Example:
        >>> x = torch.randn((1, 1), requires_grad=True)
        >>> with torch.autograd.profiler.profile() as prof:
        ...     y = x ** 2
        ...     with torch.autograd.profiler.record_function(\"label-z\"): # label the block
        ...         z = y ** 3
        ...     y.backward()
        ...
        >>> # NOTE: some columns were removed for brevity
        >>> print(prof.key_averages().table(sort_by=\"self_cpu_time_total\"))
        -----------------------------------  ---------------  ---------------  ---------------
        Name                                 Self CPU total %  CPU time avg     Number of Calls
        -----------------------------------  ---------------  ---------------  ---------------
        pow                                  60.77%           47.470us         3
        mul                                  21.73%           25.465us         2
        PowBackward0                         12.03%           121.891us        1
        torch::autograd::AccumulateGrad      2.70%            6.324us          1
        label-z                              2.13%            12.421us         1
        torch::autograd::GraphRoot           0.64%            1.503us          1
        -----------------------------------  ---------------  ---------------  ---------------
        Self CPU time total: 234.344us
        CUDA time total: 0.000us
    "
  [ name ]
  (py/call-attr profiler "record_function"  name ))
