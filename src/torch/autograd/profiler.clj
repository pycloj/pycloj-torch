(ns torch.autograd.profiler
  ""
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce profiler (import-module "torch.autograd.profiler"))

(defn attr-formatter 
  ""
  [ name ]
  (py/call-attr profiler "attr_formatter"  name ))

(defn build-table 
  "Prints a summary of events (which can be a list of FunctionEvent or FunctionEventAvg)."
  [events sort_by header & {:keys [row_limit]
                       :or {row_limit 100}} ]
    (py/call-attr-kw profiler "build_table" [events sort_by header] {:row_limit row_limit }))

(defn format-time 
  "Defines how to format time in FunctionEvent"
  [ time_us ]
  (py/call-attr profiler "format_time"  time_us ))

(defn format-time-share 
  "Defines how to format time in FunctionEvent"
  [ time_us total_time_us ]
  (py/call-attr profiler "format_time_share"  time_us total_time_us ))

(defn load-nvprof 
  "Opens an nvprof trace file and parses autograd annotations.

    Arguments:
        path (str): path to nvprof trace
    "
  [ path ]
  (py/call-attr profiler "load_nvprof"  path ))

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
    (py/call-attr-kw profiler "namedtuple" [typename field_names] {:rename rename :defaults defaults :module module }))

(defn parse-cpu-trace 
  ""
  [ thread_records ]
  (py/call-attr profiler "parse_cpu_trace"  thread_records ))

(defn parse-nvprof-trace 
  ""
  [ path ]
  (py/call-attr profiler "parse_nvprof_trace"  path ))
