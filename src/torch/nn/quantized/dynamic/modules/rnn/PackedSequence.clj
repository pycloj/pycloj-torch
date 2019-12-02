(ns torch.nn.quantized.dynamic.modules.rnn.PackedSequence
  "Holds the data and list of :attr:`batch_sizes` of a packed sequence.

    All RNN modules accept packed sequences as inputs.

    Note:
        Instances of this class should never be created manually. They are meant
        to be instantiated by functions like :func:`pack_padded_sequence`.

        Batch sizes represent the number elements at each sequence step in
        the batch, not the varying sequence lengths passed to
        :func:`pack_padded_sequence`.  For instance, given data ``abc`` and ``x``
        the :class:`PackedSequence` would contain data ``axbc`` with
        ``batch_sizes=[2,1,1]``.

    Attributes:
        data (Tensor): Tensor containing packed sequence
        batch_sizes (Tensor): Tensor of integers holding
            information about the batch size at each sequence step
        sorted_indices (Tensor, optional): Tensor of integers holding how this
            :class:`PackedSequence` is constructed from sequences.
        unsorted_indices (Tensor, optional): Tensor of integers holding how this
            to recover the original sequences with correct order.

    .. note::
        :attr:`data` can be on arbitrary device and of arbitrary dtype.
        :attr:`sorted_indices` and :attr:`unsorted_indices` must be ``torch.int64``
        tensors on the same device as :attr:`data`.

        However, :attr:`batch_sizes` should always be a CPU ``torch.int64`` tensor.

        This invariant is maintained throughout :class:`PackedSequence` class,
        and all functions that construct a `:class:PackedSequence` in PyTorch
        (i.e., they only pass in tensors conforming to this constraint).

    "
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce rnn (import-module "torch.nn.quantized.dynamic.modules.rnn"))

(defn PackedSequence 
  "Holds the data and list of :attr:`batch_sizes` of a packed sequence.

    All RNN modules accept packed sequences as inputs.

    Note:
        Instances of this class should never be created manually. They are meant
        to be instantiated by functions like :func:`pack_padded_sequence`.

        Batch sizes represent the number elements at each sequence step in
        the batch, not the varying sequence lengths passed to
        :func:`pack_padded_sequence`.  For instance, given data ``abc`` and ``x``
        the :class:`PackedSequence` would contain data ``axbc`` with
        ``batch_sizes=[2,1,1]``.

    Attributes:
        data (Tensor): Tensor containing packed sequence
        batch_sizes (Tensor): Tensor of integers holding
            information about the batch size at each sequence step
        sorted_indices (Tensor, optional): Tensor of integers holding how this
            :class:`PackedSequence` is constructed from sequences.
        unsorted_indices (Tensor, optional): Tensor of integers holding how this
            to recover the original sequences with correct order.

    .. note::
        :attr:`data` can be on arbitrary device and of arbitrary dtype.
        :attr:`sorted_indices` and :attr:`unsorted_indices` must be ``torch.int64``
        tensors on the same device as :attr:`data`.

        However, :attr:`batch_sizes` should always be a CPU ``torch.int64`` tensor.

        This invariant is maintained throughout :class:`PackedSequence` class,
        and all functions that construct a `:class:PackedSequence` in PyTorch
        (i.e., they only pass in tensors conforming to this constraint).

    "
  [ data batch_sizes sorted_indices unsorted_indices ]
  (py/call-attr rnn "PackedSequence"  data batch_sizes sorted_indices unsorted_indices ))

(defn batch-sizes 
  "Alias for field number 1"
  [ self ]
    (py/call-attr self "batch_sizes"))

(defn byte 
  "Returns copy with `self.data` cast to byte type"
  [ self  ]
  (py/call-attr self "byte"  self  ))

(defn char 
  "Returns copy with `self.data` cast to char type"
  [ self  ]
  (py/call-attr self "char"  self  ))

(defn cpu 
  "Returns a CPU copy if `self.data` not already on the CPU"
  [ self  ]
  (py/call-attr self "cpu"  self  ))

(defn cuda 
  "Returns a GPU copy if `self.data` not already on the GPU"
  [ self  ]
  (py/call-attr self "cuda"  self  ))

(defn data 
  "Alias for field number 0"
  [ self ]
    (py/call-attr self "data"))

(defn double 
  "Returns copy with `self.data` cast to double type"
  [ self  ]
  (py/call-attr self "double"  self  ))

(defn float 
  "Returns copy with `self.data` cast to float type"
  [ self  ]
  (py/call-attr self "float"  self  ))

(defn half 
  "Returns copy with `self.data` cast to half type"
  [ self  ]
  (py/call-attr self "half"  self  ))

(defn int 
  "Returns copy with `self.data` cast to int type"
  [ self  ]
  (py/call-attr self "int"  self  ))

(defn is-cuda 
  "Returns true if `self.data` stored on a gpu"
  [ self ]
    (py/call-attr self "is_cuda"))

(defn is-pinned 
  "Returns true if `self.data` stored on in pinned memory"
  [ self  ]
  (py/call-attr self "is_pinned"  self  ))

(defn long 
  "Returns copy with `self.data` cast to long type"
  [ self  ]
  (py/call-attr self "long"  self  ))

(defn pin-memory 
  ""
  [ self  ]
  (py/call-attr self "pin_memory"  self  ))

(defn short 
  "Returns copy with `self.data` cast to short type"
  [ self  ]
  (py/call-attr self "short"  self  ))

(defn sorted-indices 
  "Alias for field number 2"
  [ self ]
    (py/call-attr self "sorted_indices"))

(defn to 
  "Performs dtype and/or device conversion on `self.data`.

        It has similar signature as :meth:`torch.Tensor.to`.

        .. note::

            If the ``self.data`` Tensor already has the correct :class:`torch.dtype`
            and :class:`torch.device`, then ``self`` is returned.
            Otherwise, returns a copy with the desired configuration.
        "
  [ self  ]
  (py/call-attr self "to"  self  ))

(defn unsorted-indices 
  "Alias for field number 3"
  [ self ]
    (py/call-attr self "unsorted_indices"))
