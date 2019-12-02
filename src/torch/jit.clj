(ns torch.jit
  ""
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce jit (import-module "torch.jit"))

(defn annotate 
  ""
  [ the_type the_value ]
  (py/call-attr jit "annotate"  the_type the_value ))

(defn export 
  "
    This decorator indicates that a method is used as an entry point into a
    ``ScriptModule`` and should be compiled. ``forward`` implicitly is assumbed to be an
    entry point, so it does not need this decorator. Functions and methods
    called from ``forward`` are compiled as they are seen, so they do not need
    this decorator either.

    Example (using ``@torch.jit.export`` on a method):

    .. testcode::

        import torch
        import torch.nn as nn

        class MyModule(nn.Module):
            def implicitly_compiled_method(self, x):
                return x + 99

            # `forward` is implicitly decorated with `@torch.jit.export`,
            # so adding it here would have no effect
            def forward(self, x):
                return x + 10

            @torch.jit.export
            def another_forward(self, x):
                # When the compiler sees this call, it will compile
                # `implicitly_compiled_method`
                return self.implicitly_compiled_method(x)

            def unused_method(self, x):
                return x - 20

        # `m` will contain compiled methods:
        #     `forward`
        #     `another_forward`
        #     `implicitly_compiled_method`
        # `unused_method` will not be compiled since it was not called from
        # any compiled methods and wasn't decorated with `@torch.jit.export`
        m = torch.jit.script(MyModule())
    "
  [ fn ]
  (py/call-attr jit "export"  fn ))

(defn get-default-args 
  ""
  [ fn ]
  (py/call-attr jit "get_default_args"  fn ))

(defn get-jit-class-def 
  ""
  [ cls self_name ]
  (py/call-attr jit "get_jit_class_def"  cls self_name ))

(defn get-jit-def 
  ""
  [ fn self_name ]
  (py/call-attr jit "get_jit_def"  fn self_name ))

(defn get-trace-graph 
  "
    Trace a function or model, returning a tuple consisting of the both the
    *trace* of an execution, as well as the original return value. If return_inputs,
    also returns the trace inputs as part of the tuple

    Tracing is guaranteed not to change the semantics of the function/module
    that is traced.

    Arguments:
        f (torch.nn.Module or function): the function or module
            to be traced.
        args (tuple or Tensor): the positional arguments to pass to the
            function/module to be traced.  A non-tuple is assumed to
            be a single positional argument to be passed to the model.
        kwargs (dict): the keyword arguments to pass to the function/module
            to be traced.

    Example (trace a cell):

    .. testcode::

        trace = torch.jit.trace(nn.LSTMCell(), (input, hidden))
    "
  [f & {:keys [args kwargs _force_outplace return_inputs _return_inputs_states]
                       :or {args () _force_outplace false return_inputs false _return_inputs_states false}} ]
    (py/call-attr-kw jit "get_trace_graph" [f] {:args args :kwargs kwargs :_force_outplace _force_outplace :return_inputs return_inputs :_return_inputs_states _return_inputs_states }))

(defn ignore 
  "
    This decorator indicates to the compiler that a function or method should
    be ignored and left as a Python function. This allows you to leave code in
    your model that is not yet TorchScript compatible. Models with ignored
    functions cannot be exported; use torch.jit.unused instead.

    Example (using ``@torch.jit.ignore`` on a method)::

        import torch
        import torch.nn as nn

        class MyModule(nn.Module):
            @torch.jit.ignore
            def debugger(self, x):
                import pdb
                pdb.set_trace()

            def forward(self, x):
                x += 10
                # The compiler would normally try to compile `debugger`,
                # but since it is `@ignore`d, it will be left as a call
                # to Python
                self.debugger(x)
                return x

        m = torch.jit.script(MyModule())

        # Error! The call `debugger` cannot be saved since it calls into Python
        m.save(\"m.pt\")

    Example (using ``@torch.jit.ignore(drop=True)`` on a method):

    .. testcode::

        import torch
        import torch.nn as nn

        class MyModule(nn.Module):
            @torch.jit.ignore(drop=True)
            def training_method(self, x):
                import pdb
                pdb.set_trace()

            def forward(self, x):
                if self.training:
                    self.training_method(x)
                return x

        m = torch.jit.script(MyModule())

        # This is OK since `training_method` is not saved, the call is replaced
        # with a `raise`.
        m.save(\"m.pt\")

    .. testcleanup::

        import os
        os.remove('m.pt')
    "
  [ & {:keys [drop]
       :or {drop false}} ]
  
   (py/call-attr-kw jit "ignore" [] {:drop drop }))

(defn indent 
  ""
  [ s ]
  (py/call-attr jit "indent"  s ))

(defn interface 
  ""
  [ obj ]
  (py/call-attr jit "interface"  obj ))

(defn is-scripting 
  "
    Function that returns True when in compilation and False otherwise. This
    is useful especially with the @unused decorator to leave code in your
    model that is not yet TorchScript compatible.

    @torch.jit.unused
    def unsupported_linear_op(x):
        return x

    def linear(x):
       if not torch.jit.is_scripting():
          return torch.linear(x)
       else:
          return unsupported_linear_op(x)
    "
  [  ]
  (py/call-attr jit "is_scripting"  ))

(defn load 
  "
        Load a ``ScriptModule`` previously saved with :func:`torch.jit.save <torch.jit.save>`

        All previously saved modules, no matter their device, are first loaded onto CPU,
        and then are moved to the devices they were saved from. If this fails (e.g. because
        the run time system doesn't have certain devices), an exception is raised.

        Arguments:
            f: a file-like object (has to implement read, readline, tell, and seek),
                or a string containing a file name
            map_location (string or torch.device): A simplified version of ``map_location`` in
                ``torch.save`` used to dynamically remap storages to an alternative set of devices.
            _extra_files (dictionary of filename to content): The extra
                filenames given in the map would be loaded and their content
                would be stored in the provided map.

        Returns:
            A ``ScriptModule`` object.

        Example:

        .. testcode::

            import torch
            import io

            torch.jit.load('scriptmodule.pt')

            # Load ScriptModule from io.BytesIO object
            with open('scriptmodule.pt', 'rb') as f:
                buffer = io.BytesIO(f.read())

            # Load all tensors to the original device
            torch.jit.load(buffer)

            # Load all tensors onto CPU, using a device
            buffer.seek(0)
            torch.jit.load(buffer, map_location=torch.device('cpu'))

            # Load all tensors onto CPU, using a string
            buffer.seek(0)
            torch.jit.load(buffer, map_location='cpu')

            # Load with extra files.
            extra_files = torch._C.ExtraFilesMap()
            extra_files['foo.txt'] = 'bar'
            torch.jit.load('scriptmodule.pt', _extra_files=extra_files)
            print(extra_files['foo.txt'])

        .. testoutput::
            :hide:

            ...

        .. testcleanup::

            import os
            os.remove(\"scriptmodule.pt\")
    "
  [f map_location & {:keys [_extra_files]
                       :or {_extra_files ExtraFilesMap{}}} ]
    (py/call-attr-kw jit "load" [f map_location] {:_extra_files _extra_files }))

(defn make-module 
  ""
  [mod _module_class _compilation_unit & {:keys [exclude_methods]
                       :or {exclude_methods ()}} ]
    (py/call-attr-kw jit "make_module" [mod _module_class _compilation_unit] {:exclude_methods exclude_methods }))

(defn make-tuple 
  ""
  [ example_inputs ]
  (py/call-attr jit "make_tuple"  example_inputs ))

(defn method 
  "Sets gradients of all model parameters to zero."
  [  ]
  (py/call-attr jit "method"  ))

(defn namedtuple 
  "Returns a new subclass of tuple with named fields.

    >>> Point = namedtuple('Point', ['x', 'y'])
    >>> Point.__doc__                   # docstring for the new class
    'Point(x, y)'
    >>> p = Point(11, y=22)             # instantiate with positional args or keywords
    >>> p[0] + p[1]                     # indexable like a plain tuple
    33
    >>> x, y = p                        # unpack like a regular tuple
    >>> x, y
    (11, 22)
    >>> p.x + p.y                       # fields also accessible by name
    33
    >>> d = p._asdict()                 # convert to a dictionary
    >>> d['x']
    11
    >>> Point(**d)                      # convert from a dictionary
    Point(x=11, y=22)
    >>> p._replace(x=100)               # _replace() is like str.replace() but targets named fields
    Point(x=100, y=22)

    "
  [typename field_names & {:keys [rename defaults module]
                       :or {rename false}} ]
    (py/call-attr-kw jit "namedtuple" [typename field_names] {:rename rename :defaults defaults :module module }))

(defn optimized-execution 
  "
    A context manager that controls whether the JIT's executor will run
    optimizations before executing a function.
    "
  [ should_optimize ]
  (py/call-attr jit "optimized_execution"  should_optimize ))

(defn save 
  "
        Save an offline version of this module for use in a separate process. The saved
        module serializes all of the methods, submodules, parameters, and attributes of this
        module. It can be loaded into the C++ API using ``torch::jit::load(filename)`` or into the Python
        API with :func:`torch.jit.load <torch.jit.load>`.

        To be able to save a module, it must not make any calls to native Python functions.
        This means that all submodules must be subclasses of ``torch.jit.ScriptModule`` as well.

        .. DANGER::
           All modules, no matter their device, are always loaded onto the CPU during loading.
           This is different from :func:`load <torch.jit.load>`'s semantics and may change in the future.

        Arguments:
            m: a ScriptModule to save
            f: a file-like object (has to implement write and flush) or a string
               containing a file name
            _extra_files: Map from filename to contents which will be stored as part of 'f'

        .. warning::
            If you are using Python 2, ``torch.jit.save`` does NOT support ``StringIO.StringIO``
            as a valid file-like object. This is because the write method should return
            the number of bytes written; ``StringIO.write()`` does not do this.

            Please use something like ``io.BytesIO`` instead.

        Example:

        .. testcode::

            import torch
            import io

            class MyModule(torch.nn.Module):
                def forward(self, x):
                    return x + 10

            m = torch.jit.script(MyModule())

            # Save to file
            torch.jit.save(m, 'scriptmodule.pt')
            # This line is equivalent to the previous
            m.save(\"scriptmodule.pt\")

            # Save to io.BytesIO buffer
            buffer = io.BytesIO()
            torch.jit.save(m, buffer)

            # Save with extra files
            extra_files = torch._C.ExtraFilesMap()
            extra_files['foo.txt'] = 'bar'
            torch.jit.save(m, 'scriptmodule.pt', _extra_files=extra_files)
    "
  [m f & {:keys [_extra_files]
                       :or {_extra_files ExtraFilesMap{}}} ]
    (py/call-attr-kw jit "save" [m f] {:_extra_files _extra_files }))

(defn scope 
  ""
  [ scope_name ]
  (py/call-attr jit "scope"  scope_name ))

(defn script 
  "
    Scripting a function or ``nn.Module`` will inspect the source code, compile
    it as TorchScript code using the TorchScript compiler, and return a ``ScriptModule`` or
    ``torch._C.Function``. TorchScript itself is a subset of the Python language, so not all
    features in Python work, but we provide enough functionality to compute on
    tensors and do control-dependent operations. For a complete guide, see the
    `TorchScript Language Reference`_.

    ``torch.jit.script`` can be used as a function for modules and functions, and as a decorator
    ``@torch.jit.script`` for `TorchScript Classes <TorchScript Class_>`_ and functions.

    **Scripting a function**
        The ``@torch.jit.script`` decorator will construct a ``torch._C.Function``
        by compiling the body of the function.

        Example (scripting a function):

        .. testcode::

            import torch

            @torch.jit.script
            def foo(x, y):
                if x.max() > y.max():
                    r = x
                else:
                    r = y
                return r

    **Scripting an nn.Module**
        Scripting an ``nn.Module`` by default will compile the ``forward`` method and recursively
        compile any methods, submodules, and functions called by ``forward``. If a ``nn.Module`` only uses
        features supported in TorchScript, no changes to the original module code should be necessary. ``script``
        will construct ``torch.jit.ScriptModule`` that has copies of the attributes, parameters, and methods of
        the original module.

        Example (scripting a simple module with a Parameter):

        .. testcode::

            import torch

            class MyModule(torch.nn.Module):
                def __init__(self, N, M):
                    super(MyModule, self).__init__()
                    # This parameter will be copied to the new ScriptModule
                    self.weight = torch.nn.Parameter(torch.rand(N, M))

                    # When this submodule is used, it will be compiled
                    self.linear = torch.nn.Linear(N, M)

                def forward(self, input):
                    output = self.weight.mv(input)

                    # This calls the `forward` method of the `nn.Linear` module, which will
                    # cause the `self.linear` submodule to be compiled to a `ScriptModule` here
                    output = self.linear(output)
                    return output

            scripted_module = torch.jit.script(MyModule(2, 3))

        Example (scripting a module with traced submodules):

        .. testcode::

            import torch
            import torch.nn as nn
            import torch.nn.functional as F

            class MyModule(nn.Module):
                def __init__(self):
                    super(MyModule, self).__init__()
                    # torch.jit.trace produces a ScriptModule's conv1 and conv2
                    self.conv1 = torch.jit.trace(nn.Conv2d(1, 20, 5), torch.rand(1, 1, 16, 16))
                    self.conv2 = torch.jit.trace(nn.Conv2d(20, 20, 5), torch.rand(1, 20, 16, 16))

                def forward(self, input):
                  input = F.relu(self.conv1(input))
                  input = F.relu(self.conv2(input))
                  return input

            scripted_module = torch.jit.script(MyModule())

        To compile a method other than ``forward`` (and recursively compile anything it calls), add
        the :func:`@torch.jit.export <torch.jit.export>` decorator to the method. To opt out of compilation
        use :func:`@torch.jit.ignore <torch.jit.ignore>`.

        Example (an exported and ignored method in a module)::

            import torch
            import torch.nn as nn

            class MyModule(nn.Module):
                def __init__(self):
                    super(MyModule, self).__init__()

                @torch.jit.export
                def some_entry_point(self, input):
                    return input + 10

                @torch.jit.ignore
                def python_only_fn(self, input):
                    # This function won't be compiled, so any
                    # Python APIs can be used
                    import pdb
                    pdb.set_trace()

                def forward(self, input):
                    if self.training:
                        self.python_only_fn(input)
                    return input * 99

            scripted_module = torch.jit.script(MyModule())
            print(scripted_module.some_entry_point(torch.randn(2, 2)))
            print(scripted_module(torch.randn(2, 2)))
    "
  [obj optimize & {:keys [_frames_up _rcb]
                       :or {_frames_up 0}} ]
    (py/call-attr-kw jit "script" [obj optimize] {:_frames_up _frames_up :_rcb _rcb }))

(defn script-method 
  ""
  [ fn _rcb ]
  (py/call-attr jit "script_method"  fn _rcb ))

(defn trace 
  "
    Trace a function and return an executable ``ScriptModule`` or ``torch._C.Function``
    that will be optimized using just-in-time compilation.

    Using ``torch.jit.trace`` and :func:`torch.jit.trace_module<torch.jit.trace_module>`, you can turn an existing module or Python
    function into a TorchScript ``torch._C.Function`` or ``ScriptModule``. You must provide example inputs,
    and we run the function, recording the operations performed on all the tensors.

    * The resulting recording of a standalone function produces ``torch._C.Function``.
    * The resulting recording of ``forward`` function of ``nn.Module`` or ``nn.Module`` produces ``ScriptModule``.

    This module also contains any parameters that the original
    module had as well.

    .. warning::
        Tracing only correctly records functions and modules which are not data
        dependent (e.g., do not have conditionals on data in tensors) and do not have
        any untracked external dependencies (e.g., perform input/output or
        access global variables). Tracing only records operations done when the given
        function is run on the given
        tensors. Therefore, the returned ``ScriptModule`` will always run the same traced
        graph on any input. This has some important implications when your module is
        expected to run different sets of operations, depending on the input and/or the
        module state. For example,

        * Tracing will not record any control-flow like if-statements or loops.
          When this control-flow is constant across your module, this is fine and it often
          inlines the control-flow decisions. But sometimes the control-flow is actually part
          of the model itself. For instance, a recurrent network is a loop over
          the (possibly dynamic) length of an input sequence.
        * In the returned ``ScriptModule``, operations that have different
          behaviors in ``training`` and ``eval`` modes will always behave as if it
          is in the mode it was in during tracing, no matter which mode the
          ``ScriptModule`` is in.

        In cases like these, tracing would not be appropriate and :func:`scripting <torch.jit.script>` is a better
        choice. If you trace such models, you may silently get
        incorrect results on subsequent invocations of the model. The tracer
        will try to emit warnings when doing something that may cause an
        incorrect trace to be produced.

    Arguments:
        func (callable or torch.nn.Module):  a Python function or ``torch.nn.Module``
                                             that will be run with ``example_inputs``.
                                             arguments and returns to ``func`` must be tensors
                                             or (possibly nested) tuples that
                                             contain tensors.
        example_inputs (tuple):  a tuple of example inputs that will be passed to the function
                                 while tracing. The resulting trace can be run with
                                 inputs of different types and shapes assuming the traced operations
                                 support those types and shapes. ``example_inputs`` may also be a single
                                 Tensor in which case it is automatically wrapped in a tuple

    Keyword arguments:
        check_trace (bool, optional): check if the same inputs run through
                                      traced code produce the same outputs. Default: ``True``. You might want
                                      to disable this if, for example, your network contains non-
                                      deterministic ops or if you are sure that the network is correct despite
                                      a checker failure.

        check_inputs (list of tuples, optional): A list of tuples of input arguments that should be used
                                                 to check the trace against what is expected. Each tuple
                                                 is equivalent to a set of input arguments that would
                                                 be specified in ``example_inputs``. For best results, pass in a
                                                 set of checking inputs representative of the space of
                                                 shapes and types of inputs you expect the network to see.
                                                 If not specified, the original ``example_inputs`` are used for checking
        check_tolerance (float, optional): Floating-point comparison tolerance to use in the checker procedure.
                                           This can be used to relax the checker strictness in the event that
                                           results diverge numerically for a known reason, such as operator fusion.

    Returns:
        if ``callable`` is ``nn.Module`` or ``forward()`` of ``nn.Module``, ``trace`` returns
        a ``ScriptModule`` object with a single ``forward()`` method containing the traced code.
        The returned ``ScriptModule`` will have the same set of sub-modules and parameters as the
        original ``nn.Module``.
        If ``callable`` is a standalone function, ``trace`` returns ``torch._C.Function``

    Example (tracing a function):

    .. testcode::

        import torch

        def foo(x, y):
            return 2 * x + y

        # Run `foo` with the provided inputs and record the tensor operations
        traced_foo = torch.jit.trace(foo, (torch.rand(3), torch.rand(3)))

        # `traced_foo` can now be run with the TorchScript interpreter or saved
        # and loaded in a Python-free environment

    Example (tracing an existing module)::

        import torch
        import torch.nn as nn

        class Net(nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                self.conv = nn.Conv2d(1, 1, 3)

            def forward(self, x):
                return self.conv(x)

        n = Net()
        example_weight = torch.rand(1, 1, 3, 3)
        example_forward_input = torch.rand(1, 1, 3, 3)

        # Trace a specific method and construct `ScriptModule` with
        # a single `forward` method
        module = torch.jit.trace(n.forward, example_forward_input)

        # Trace a module (implicitly traces `forward`) and construct a
        # `ScriptModule` with a single `forward` method
        module = torch.jit.trace(n, example_forward_input)
    "
  [func example_inputs optimize & {:keys [check_trace check_inputs check_tolerance _force_outplace _module_class _compilation_unit]
                       :or {check_trace true check_tolerance 1e-05 _force_outplace false _compilation_unit <torch._C.CompilationUnit object at 0x121b8c7f0>}} ]
    (py/call-attr-kw jit "trace" [func example_inputs optimize] {:check_trace check_trace :check_inputs check_inputs :check_tolerance check_tolerance :_force_outplace _force_outplace :_module_class _module_class :_compilation_unit _compilation_unit }))

(defn trace-module 
  "
    Trace a module and return an executable ``ScriptModule`` that will be optimized
    using just-in-time compilation. When a module is passed to :func:`torch.jit.trace <torch.jit.trace>`, only
    the ``forward`` method is run and traced. With ``trace_module``, you can specify a dictionary of
    method names to example inputs to trace (see the ``example_inputs``) argument below.

    See :func:`torch.jit.trace <torch.jit.trace>` for more information on tracing.

    Arguments:
        mod (torch.nn.Module):           a ``torch.nn.Module`` containing methods whose names are
                                         specified in ``example_inputs``. The given methods will be compiled
                                         as a part of a single `ScriptModule`
        example_inputs (dict):           a dict containing sample inputs indexed by method names in ``mod``
                                         The inputs will be passed to methods whose names correspond to inputs'
                                         keys while tracing.
                                         ``{ 'forward' : example_forward_input, 'method2': example_method2_input}``
    Keyword arguments:
        check_trace (bool, optional): check if the same inputs run through
                                      traced code produce the same outputs. Default: ``True``. You might want
                                      to disable this if, for example, your network contains non-
                                      deterministic ops or if you are sure that the network is correct despite
                                      a checker failure.

        check_inputs (list of dicts, optional): A list of dicts of input arguments that should be used
                                                 to check the trace against what is expected. Each tuple
                                                 is equivalent to a set of input arguments that would
                                                 be specified in ``example_inputs``. For best results, pass in a
                                                 set of checking inputs representative of the space of
                                                 shapes and types of inputs you expect the network to see.
                                                 If not specified, the original ``example_inputs`` are used for checking
        check_tolerance (float, optional): Floating-point comparison tolerance to use in the checker procedure.
                                           This can be used to relax the checker strictness in the event that
                                           results diverge numerically for a known reason, such as operator fusion.

    Returns:
        A ``ScriptModule`` object with a single ``forward()`` method containing the traced code.
        When ``func`` is a ``torch.nn.Module``, the returned ``ScriptModule`` will have the same set of
        sub-modules and parameters as ``func``.

    Example (tracing a module with multiple methods)::

        import torch
        import torch.nn as nn

        class Net(nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                self.conv = nn.Conv2d(1, 1, 3)

            def forward(self, x):
                return self.conv(x)

            def weighted_kernel_sum(self, weight):
                return weight * self.conv.weight


        n = Net()
        example_weight = torch.rand(1, 1, 3, 3)
        example_forward_input = torch.rand(1, 1, 3, 3)

        # Trace a specific method and construct `ScriptModule` with
        # a single `forward` method
        module = torch.jit.trace(n.forward, example_forward_input)

        # Trace a module (implicitly traces `forward`) and construct a
        # `ScriptModule` with a single `forward` method
        module = torch.jit.trace(n, example_forward_input)

        # Trace specific methods on a module (specified in `inputs`), constructs
        # a `ScriptModule` with `forward` and `weighted_kernel_sum` methods
        inputs = {'forward' : example_forward_input, 'weighted_kernel_sum' : example_weight}
        module = torch.jit.trace_module(n, inputs)
    "
  [mod inputs optimize & {:keys [check_trace check_inputs check_tolerance _force_outplace _module_class _compilation_unit]
                       :or {check_trace true check_tolerance 1e-05 _force_outplace false _compilation_unit <torch._C.CompilationUnit object at 0x121b8c7f0>}} ]
    (py/call-attr-kw jit "trace_module" [mod inputs optimize] {:check_trace check_trace :check_inputs check_inputs :check_tolerance check_tolerance :_force_outplace _force_outplace :_module_class _module_class :_compilation_unit _compilation_unit }))

(defn unused 
  "
    This decorator indicates to the compiler that a function or method should
    be ignored and replaced with the raising of an exception. This allows you
    to leave code in your model that is not yet TorchScript compatible and still
    export your model.

        Example (using ``@torch.jit.unused`` on a method)::

            import torch
            import torch.nn as nn

            class MyModule(nn.Module):
                def __init__(self, use_memory_efficent):
                    super(MyModule, self).__init__()
                    self.use_memory_efficent = use_memory_efficent

                @torch.jit.unused
                def memory_efficient(self, x):
                    import pdb
                    pdb.set_trace()
                    return x + 10

                def forward(self, x):
                    # Use not-yet-scriptable memory efficient mode
                    if self.use_memory_efficient:
                        return self.memory_efficient(x)
                    else:
                        return x + 10

            m = torch.jit.script(MyModule(use_memory_efficent=False))
            m.save(\"m.pt\")

            m = torch.jit.script(MyModule(use_memory_efficient=True))
            # exception raised
            m(torch.rand(100))
    "
  [ fn ]
  (py/call-attr jit "unused"  fn ))

(defn validate-cuda-device 
  ""
  [ location ]
  (py/call-attr jit "validate_cuda_device"  location ))

(defn verify 
  "
    Verify that a JIT compiled model has the same behavior as its uncompiled
    version along with its backwards pass.  If your model returns multiple
    outputs, you must also specify a `loss_fn` to produce a loss for which
    the backwards will be computed.

    This function has side-effects (e.g., it executes your model / saves and loads
    parameters), so don't expect the model to come out exactly the same as what
    you passed in.

    Arguments:
        model (compiled torch.nn.Module or function): the module/function to be
            verified.  The module/function definition MUST have been decorated with
            `@torch.jit.compile`.
        args (tuple or Tensor): the positional arguments to pass to the
            compiled function/module to be verified.  A non-tuple is assumed to
            be a single positional argument to be passed to the model.
        loss_fn (function, optional): the loss function to be applied to
            the output of the model, before backwards is invoked.  By default,
            we assume that a model returns a single result, and we :func:`torch.sum`
            before calling backwards; if this is inappropriate, you can pass your
            own loss function.  Note that if a model returns a tuple of results,
            these are passed as separate positional arguments to `loss_fn`.
        devices (iterable of device IDs, optional): the GPU devices which the
            compiled module will be run on.  This determines the RNG state we
            must save when running both compiled and uncompiled versions of the model.
    "
  [model args & {:keys [loss_fn devices]
                       :or {loss_fn <built-in method sum of type object at 0x11b3c4fd0>}} ]
    (py/call-attr-kw jit "verify" [model args] {:loss_fn loss_fn :devices devices }))

(defn whichmodule 
  "Find the module an object belong to."
  [ obj ]
  (py/call-attr jit "whichmodule"  obj ))

(defn with-metaclass 
  "Create a base class with a metaclass."
  [ meta ]
  (py/call-attr jit "with_metaclass"  meta ))

(defn wrap-check-inputs 
  ""
  [ check_inputs ]
  (py/call-attr jit "wrap_check_inputs"  check_inputs ))
