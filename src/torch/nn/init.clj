(ns torch.nn.init
  ""
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce init (import-module "torch.nn.init"))

(defn calculate-gain 
  "Return the recommended gain value for the given nonlinearity function.
    The values are as follows:

    ================= ====================================================
    nonlinearity      gain
    ================= ====================================================
    Linear / Identity :math:`1`
    Conv{1,2,3}D      :math:`1`
    Sigmoid           :math:`1`
    Tanh              :math:`\frac{5}{3}`
    ReLU              :math:`\sqrt{2}`
    Leaky Relu        :math:`\sqrt{\frac{2}{1 + \text{negative\_slope}^2}}`
    ================= ====================================================

    Args:
        nonlinearity: the non-linear function (`nn.functional` name)
        param: optional parameter for the non-linear function

    Examples:
        >>> gain = nn.init.calculate_gain('leaky_relu', 0.2)  # leaky_relu with negative_slope=0.2
    "
  [ nonlinearity param ]
  (py/call-attr init "calculate_gain"  nonlinearity param ))

(defn constant 
  "
    constant(...)

    .. warning::
        This method is now deprecated in favor of :func:`torch.nn.init.constant_`.

    See :func:`~torch.nn.init.constant_` for details."
  [  ]
  (py/call-attr init "constant"  ))

(defn constant- 
  "Fills the input Tensor with the value :math:`\text{val}`.

    Args:
        tensor: an n-dimensional `torch.Tensor`
        val: the value to fill the tensor with

    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.constant_(w, 0.3)
    "
  [ tensor val ]
  (py/call-attr init "constant_"  tensor val ))

(defn dirac 
  "
    dirac(...)

    .. warning::
        This method is now deprecated in favor of :func:`torch.nn.init.dirac_`.

    See :func:`~torch.nn.init.dirac_` for details."
  [  ]
  (py/call-attr init "dirac"  ))

(defn dirac- 
  "Fills the {3, 4, 5}-dimensional input `Tensor` with the Dirac
    delta function. Preserves the identity of the inputs in `Convolutional`
    layers, where as many input channels are preserved as possible.

    Args:
        tensor: a {3, 4, 5}-dimensional `torch.Tensor`

    Examples:
        >>> w = torch.empty(3, 16, 5, 5)
        >>> nn.init.dirac_(w)
    "
  [ tensor ]
  (py/call-attr init "dirac_"  tensor ))

(defn eye 
  "
    eye(...)

    .. warning::
        This method is now deprecated in favor of :func:`torch.nn.init.eye_`.

    See :func:`~torch.nn.init.eye_` for details."
  [  ]
  (py/call-attr init "eye"  ))

(defn eye- 
  "Fills the 2-dimensional input `Tensor` with the identity
    matrix. Preserves the identity of the inputs in `Linear` layers, where as
    many inputs are preserved as possible.

    Args:
        tensor: a 2-dimensional `torch.Tensor`

    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.eye_(w)
    "
  [ tensor ]
  (py/call-attr init "eye_"  tensor ))

(defn kaiming-normal 
  "
    kaiming_normal(...)

    .. warning::
        This method is now deprecated in favor of :func:`torch.nn.init.kaiming_normal_`.

    See :func:`~torch.nn.init.kaiming_normal_` for details."
  [  ]
  (py/call-attr init "kaiming_normal"  ))

(defn kaiming-normal- 
  "Fills the input `Tensor` with values according to the method
    described in `Delving deep into rectifiers: Surpassing human-level
    performance on ImageNet classification` - He, K. et al. (2015), using a
    normal distribution. The resulting tensor will have values sampled from
    :math:`\mathcal{N}(0, \text{std}^2)` where

    .. math::
        \text{std} = \sqrt{\frac{2}{(1 + a^2) \times \text{fan\_in}}}

    Also known as He initialization.

    Args:
        tensor: an n-dimensional `torch.Tensor`
        a: the negative slope of the rectifier used after this layer (0 for ReLU
            by default)
        mode: either ``'fan_in'`` (default) or ``'fan_out'``. Choosing ``'fan_in'``
            preserves the magnitude of the variance of the weights in the
            forward pass. Choosing ``'fan_out'`` preserves the magnitudes in the
            backwards pass.
        nonlinearity: the non-linear function (`nn.functional` name),
            recommended to use only with ``'relu'`` or ``'leaky_relu'`` (default).

    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.kaiming_normal_(w, mode='fan_out', nonlinearity='relu')
    "
  [tensor & {:keys [a mode nonlinearity]
                       :or {a 0 mode "fan_in" nonlinearity "leaky_relu"}} ]
    (py/call-attr-kw init "kaiming_normal_" [tensor] {:a a :mode mode :nonlinearity nonlinearity }))

(defn kaiming-uniform 
  "
    kaiming_uniform(...)

    .. warning::
        This method is now deprecated in favor of :func:`torch.nn.init.kaiming_uniform_`.

    See :func:`~torch.nn.init.kaiming_uniform_` for details."
  [  ]
  (py/call-attr init "kaiming_uniform"  ))

(defn kaiming-uniform- 
  "Fills the input `Tensor` with values according to the method
    described in `Delving deep into rectifiers: Surpassing human-level
    performance on ImageNet classification` - He, K. et al. (2015), using a
    uniform distribution. The resulting tensor will have values sampled from
    :math:`\mathcal{U}(-\text{bound}, \text{bound})` where

    .. math::
        \text{bound} = \sqrt{\frac{6}{(1 + a^2) \times \text{fan\_in}}}

    Also known as He initialization.

    Args:
        tensor: an n-dimensional `torch.Tensor`
        a: the negative slope of the rectifier used after this layer (0 for ReLU
            by default)
        mode: either ``'fan_in'`` (default) or ``'fan_out'``. Choosing ``'fan_in'``
            preserves the magnitude of the variance of the weights in the
            forward pass. Choosing ``'fan_out'`` preserves the magnitudes in the
            backwards pass.
        nonlinearity: the non-linear function (`nn.functional` name),
            recommended to use only with ``'relu'`` or ``'leaky_relu'`` (default).

    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.kaiming_uniform_(w, mode='fan_in', nonlinearity='relu')
    "
  [tensor & {:keys [a mode nonlinearity]
                       :or {a 0 mode "fan_in" nonlinearity "leaky_relu"}} ]
    (py/call-attr-kw init "kaiming_uniform_" [tensor] {:a a :mode mode :nonlinearity nonlinearity }))

(defn normal 
  "
    normal(...)

    .. warning::
        This method is now deprecated in favor of :func:`torch.nn.init.normal_`.

    See :func:`~torch.nn.init.normal_` for details."
  [  ]
  (py/call-attr init "normal"  ))

(defn normal- 
  "Fills the input Tensor with values drawn from the normal
    distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`.

    Args:
        tensor: an n-dimensional `torch.Tensor`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution

    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.normal_(w)
    "
  [tensor & {:keys [mean std]
                       :or {mean 0.0 std 1.0}} ]
    (py/call-attr-kw init "normal_" [tensor] {:mean mean :std std }))

(defn ones- 
  "Fills the input Tensor with the scalar value `1`.

    Args:
        tensor: an n-dimensional `torch.Tensor`

    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.ones_(w)
    "
  [ tensor ]
  (py/call-attr init "ones_"  tensor ))

(defn orthogonal 
  "
    orthogonal(...)

    .. warning::
        This method is now deprecated in favor of :func:`torch.nn.init.orthogonal_`.

    See :func:`~torch.nn.init.orthogonal_` for details."
  [  ]
  (py/call-attr init "orthogonal"  ))

(defn orthogonal- 
  "Fills the input `Tensor` with a (semi) orthogonal matrix, as
    described in `Exact solutions to the nonlinear dynamics of learning in deep
    linear neural networks` - Saxe, A. et al. (2013). The input tensor must have
    at least 2 dimensions, and for tensors with more than 2 dimensions the
    trailing dimensions are flattened.

    Args:
        tensor: an n-dimensional `torch.Tensor`, where :math:`n \geq 2`
        gain: optional scaling factor

    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.orthogonal_(w)
    "
  [tensor & {:keys [gain]
                       :or {gain 1}} ]
    (py/call-attr-kw init "orthogonal_" [tensor] {:gain gain }))

(defn sparse 
  "
    sparse(...)

    .. warning::
        This method is now deprecated in favor of :func:`torch.nn.init.sparse_`.

    See :func:`~torch.nn.init.sparse_` for details."
  [  ]
  (py/call-attr init "sparse"  ))

(defn sparse- 
  "Fills the 2D input `Tensor` as a sparse matrix, where the
    non-zero elements will be drawn from the normal distribution
    :math:`\mathcal{N}(0, 0.01)`, as described in `Deep learning via
    Hessian-free optimization` - Martens, J. (2010).

    Args:
        tensor: an n-dimensional `torch.Tensor`
        sparsity: The fraction of elements in each column to be set to zero
        std: the standard deviation of the normal distribution used to generate
            the non-zero values

    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.sparse_(w, sparsity=0.1)
    "
  [tensor sparsity & {:keys [std]
                       :or {std 0.01}} ]
    (py/call-attr-kw init "sparse_" [tensor sparsity] {:std std }))

(defn uniform 
  "
    uniform(...)

    .. warning::
        This method is now deprecated in favor of :func:`torch.nn.init.uniform_`.

    See :func:`~torch.nn.init.uniform_` for details."
  [  ]
  (py/call-attr init "uniform"  ))

(defn uniform- 
  "Fills the input Tensor with values drawn from the uniform
    distribution :math:`\mathcal{U}(a, b)`.

    Args:
        tensor: an n-dimensional `torch.Tensor`
        a: the lower bound of the uniform distribution
        b: the upper bound of the uniform distribution

    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.uniform_(w)
    "
  [tensor & {:keys [a b]
                       :or {a 0.0 b 1.0}} ]
    (py/call-attr-kw init "uniform_" [tensor] {:a a :b b }))

(defn xavier-normal 
  "
    xavier_normal(...)

    .. warning::
        This method is now deprecated in favor of :func:`torch.nn.init.xavier_normal_`.

    See :func:`~torch.nn.init.xavier_normal_` for details."
  [  ]
  (py/call-attr init "xavier_normal"  ))

(defn xavier-normal- 
  "Fills the input `Tensor` with values according to the method
    described in `Understanding the difficulty of training deep feedforward
    neural networks` - Glorot, X. & Bengio, Y. (2010), using a normal
    distribution. The resulting tensor will have values sampled from
    :math:`\mathcal{N}(0, \text{std}^2)` where

    .. math::
        \text{std} = \text{gain} \times \sqrt{\frac{2}{\text{fan\_in} + \text{fan\_out}}}

    Also known as Glorot initialization.

    Args:
        tensor: an n-dimensional `torch.Tensor`
        gain: an optional scaling factor

    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.xavier_normal_(w)
    "
  [tensor & {:keys [gain]
                       :or {gain 1.0}} ]
    (py/call-attr-kw init "xavier_normal_" [tensor] {:gain gain }))

(defn xavier-uniform 
  "
    xavier_uniform(...)

    .. warning::
        This method is now deprecated in favor of :func:`torch.nn.init.xavier_uniform_`.

    See :func:`~torch.nn.init.xavier_uniform_` for details."
  [  ]
  (py/call-attr init "xavier_uniform"  ))

(defn xavier-uniform- 
  "Fills the input `Tensor` with values according to the method
    described in `Understanding the difficulty of training deep feedforward
    neural networks` - Glorot, X. & Bengio, Y. (2010), using a uniform
    distribution. The resulting tensor will have values sampled from
    :math:`\mathcal{U}(-a, a)` where

    .. math::
        a = \text{gain} \times \sqrt{\frac{6}{\text{fan\_in} + \text{fan\_out}}}

    Also known as Glorot initialization.

    Args:
        tensor: an n-dimensional `torch.Tensor`
        gain: an optional scaling factor

    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.xavier_uniform_(w, gain=nn.init.calculate_gain('relu'))
    "
  [tensor & {:keys [gain]
                       :or {gain 1.0}} ]
    (py/call-attr-kw init "xavier_uniform_" [tensor] {:gain gain }))

(defn zeros- 
  "Fills the input Tensor with the scalar value `0`.

    Args:
        tensor: an n-dimensional `torch.Tensor`

    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.zeros_(w)
    "
  [ tensor ]
  (py/call-attr init "zeros_"  tensor ))
