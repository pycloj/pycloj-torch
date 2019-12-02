(ns torch.autograd.profiler.EventList
  "A list of Events (for pretty printing)"
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce profiler (import-module "torch.autograd.profiler"))

(defn EventList 
  "A list of Events (for pretty printing)"
  [  ]
  (py/call-attr profiler "EventList"  ))

(defn cpu-children-populated 
  ""
  [ self ]
    (py/call-attr self "cpu_children_populated"))

(defn export-chrome-trace 
  "Exports an EventList as a Chrome tracing tools file.

        The checkpoint can be later loaded and inspected under ``chrome://tracing`` URL.

        Arguments:
            path (str): Path where the trace will be written.
        "
  [ self path ]
  (py/call-attr self "export_chrome_trace"  self path ))

(defn key-averages 
  "Averages all function events over their keys.

        @param group_by_input_shapes The key would become
        (event name, input dimensions) rather than just event name.
        This is useful to see which dimensionality contributes to the runtime
        the most and may help with dimension specific optimizations or
        choosing best candidates for quantization (aka fitting a roof line)

        Returns:
            An EventList containing FunctionEventAvg objects.
        "
  [self  & {:keys [group_by_input_shapes]
                       :or {group_by_input_shapes false}} ]
    (py/call-attr-kw self "key_averages" [] {:group_by_input_shapes group_by_input_shapes }))

(defn populate-cpu-children 
  "Populates child events into each underlying FunctionEvent object.
        One event is a child of another if [s1, e1) is inside [s2, e2). Where
        s1 and e1 would be start and end of the child event's interval. And
        s2 and e2 start and end of the parent event's interval

        Example: In event list [[0, 10], [1, 3], [3, 4]] would have make [0, 10]
        be a parent of two other intervals.

        If for any reason two intervals intersect only partialy, this function
        will not record a parent child relationship between then.
        "
  [ self  ]
  (py/call-attr self "populate_cpu_children"  self  ))

(defn self-cpu-time-total 
  ""
  [ self ]
    (py/call-attr self "self_cpu_time_total"))

(defn table 
  "Prints an EventList as a nicely formatted table.

        Arguments:
            sort_by (str, optional): Attribute used to sort entries. By default
                they are printed in the same order as they were registered.
                Valid keys include: ``cpu_time``, ``cuda_time``, ``cpu_time_total``,
                ``cuda_time_total``, ``count``.

        Returns:
            A string containing the table.
        "
  [self sort_by & {:keys [row_limit header]
                       :or {row_limit 100}} ]
    (py/call-attr-kw self "table" [sort_by] {:row_limit row_limit :header header }))

(defn total-average 
  "Averages all events.

        Returns:
            A FunctionEventAvg object.
        "
  [ self  ]
  (py/call-attr self "total_average"  self  ))
