(ns torch.cuda.Event
  "Wrapper around a CUDA event.

    CUDA events are synchronization markers that can be used to monitor the
    device's progress, to accurately measure timing, and to synchronize CUDA
    streams.

    The underlying CUDA events are lazily initialized when the event is first
    recorded or exported to another process. After creation, only streams on the
    same device may record the event. However, streams on any device can wait on
    the event.

    Arguments:
        enable_timing (bool, optional): indicates if the event should measure time
            (default: ``False``)
        blocking (bool, optional): if ``True``, :meth:`wait` will be blocking (default: ``False``)
        interprocess (bool): if ``True``, the event can be shared between processes
            (default: ``False``)

       .. _CUDA documentation:
       https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EVENT.html
    "
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce cuda (import-module "torch.cuda"))

(defn Event 
  "Wrapper around a CUDA event.

    CUDA events are synchronization markers that can be used to monitor the
    device's progress, to accurately measure timing, and to synchronize CUDA
    streams.

    The underlying CUDA events are lazily initialized when the event is first
    recorded or exported to another process. After creation, only streams on the
    same device may record the event. However, streams on any device can wait on
    the event.

    Arguments:
        enable_timing (bool, optional): indicates if the event should measure time
            (default: ``False``)
        blocking (bool, optional): if ``True``, :meth:`wait` will be blocking (default: ``False``)
        interprocess (bool): if ``True``, the event can be shared between processes
            (default: ``False``)

       .. _CUDA documentation:
       https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EVENT.html
    "
  [ & {:keys [enable_timing blocking interprocess]
       :or {enable_timing false blocking false interprocess false}} ]
  
   (py/call-attr-kw cuda "Event" [] {:enable_timing enable_timing :blocking blocking :interprocess interprocess }))

(defn elapsed-time 
  "Returns the time elapsed in milliseconds after the event was
        recorded and before the end_event was recorded.
        "
  [ self end_event ]
  (py/call-attr self "elapsed_time"  self end_event ))

(defn ipc-handle 
  "Returns an IPC handle of this event. If not recorded yet, the event
        will use the current device. "
  [ self  ]
  (py/call-attr self "ipc_handle"  self  ))

(defn query 
  "Checks if all work currently captured by event has completed.

        Returns:
            A boolean indicating if all work currently captured by event has
            completed.
        "
  [ self  ]
  (py/call-attr self "query"  self  ))

(defn record 
  "Records the event in a given stream.

        Uses ``torch.cuda.current_stream()`` if no stream is specified. The
        stream's device must match the event's device."
  [ self stream ]
  (py/call-attr self "record"  self stream ))

(defn synchronize 
  "Waits for the event to complete.

        Waits until the completion of all work currently captured in this event.
        This prevents the CPU thread from proceeding until the event completes.

         .. note:: This is a wrapper around ``cudaEventSynchronize()``: see `CUDA
           documentation`_ for more info.

        .. _CUDA documentation:
           https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EVENT.html
        "
  [ self  ]
  (py/call-attr self "synchronize"  self  ))

(defn wait 
  "Makes all future work submitted to the given stream wait for this
        event.

        Use ``torch.cuda.current_stream()`` if no stream is specified."
  [ self stream ]
  (py/call-attr self "wait"  self stream ))
