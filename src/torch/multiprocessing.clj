(ns torch.multiprocessing
  "
torch.multiprocessing is a wrapper around the native :mod:`multiprocessing`
module. It registers custom reducers, that use shared memory to provide shared
views on the same data in different processes. Once the tensor/storage is moved
to shared_memory (see :func:`~torch.Tensor.share_memory_`), it will be possible
to send it to other processes without making any copies.

The API is 100% compatible with the original module - it's enough to change
``import multiprocessing`` to ``import torch.multiprocessing`` to have all the
tensors sent through the queues or shared via other mechanisms, moved to shared
memory.

Because of the similarity of APIs we do not document most of this package
contents, and we recommend referring to very good docs of the original module.
"
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce multiprocessing (import-module "torch.multiprocessing"))

(defn active-children 
  "
    Return list of process objects corresponding to live child processes
    "
  [  ]
  (py/call-attr multiprocessing "active_children"  ))

(defn current-process 
  "
    Return process object representing the current process
    "
  [  ]
  (py/call-attr multiprocessing "current_process"  ))

(defn get-all-sharing-strategies 
  "Returns a set of sharing strategies supported on a current system."
  [  ]
  (py/call-attr multiprocessing "get_all_sharing_strategies"  ))

(defn get-sharing-strategy 
  "Returns the current strategy for sharing CPU tensors."
  [  ]
  (py/call-attr multiprocessing "get_sharing_strategy"  ))

(defn init-reductions 
  ""
  [  ]
  (py/call-attr multiprocessing "init_reductions"  ))

(defn set-sharing-strategy 
  "Sets the strategy for sharing CPU tensors.

    Arguments:
        new_strategy (str): Name of the selected strategy. Should be one of
            the values returned by :func:`get_all_sharing_strategies()`.
    "
  [ new_strategy ]
  (py/call-attr multiprocessing "set_sharing_strategy"  new_strategy ))

(defn spawn 
  "Spawns ``nprocs`` processes that run ``fn`` with ``args``.

    If one of the processes exits with a non-zero exit status, the
    remaining processes are killed and an exception is raised with the
    cause of termination. In the case an exception was caught in the
    child process, it is forwarded and its traceback is included in
    the exception raised in the parent process.

    Arguments:
        fn (function): Function is called as the entrypoint of the
            spawned process. This function must be defined at the top
            level of a module so it can be pickled and spawned. This
            is a requirement imposed by multiprocessing.

            The function is called as ``fn(i, *args)``, where ``i`` is
            the process index and ``args`` is the passed through tuple
            of arguments.

        args (tuple): Arguments passed to ``fn``.
        nprocs (int): Number of processes to spawn.
        join (bool): Perform a blocking join on all processes.
        daemon (bool): The spawned processes' daemon flag. If set to True,
                       daemonic processes will be created.

    Returns:
        None if ``join`` is ``True``,
        :class:`~SpawnContext` if ``join`` is ``False``

    "
  [fn & {:keys [args nprocs join daemon]
                       :or {args () nprocs 1 join true daemon false}} ]
    (py/call-attr-kw multiprocessing "spawn" [fn] {:args args :nprocs nprocs :join join :daemon daemon }))
