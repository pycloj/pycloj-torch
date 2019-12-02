(ns torch.multiprocessing.Process
  ""
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce multiprocessing (import-module "torch.multiprocessing"))

(defn Process 
  ""
  [group target name & {:keys [args kwargs daemon]
                       :or {args () kwargs {}}} ]
    (py/call-attr-kw multiprocessing "Process" [group target name] {:args args :kwargs kwargs :daemon daemon }))

(defn authkey 
  ""
  [ self ]
    (py/call-attr self "authkey"))

(defn close 
  "
        Close the Process object.

        This method releases resources held by the Process object.  It is
        an error to call this method if the child process is still running.
        "
  [ self  ]
  (py/call-attr self "close"  self  ))

(defn daemon 
  "
        Return whether process is a daemon
        "
  [ self ]
    (py/call-attr self "daemon"))

(defn exitcode 
  "
        Return exit code of process or `None` if it has yet to stop
        "
  [ self ]
    (py/call-attr self "exitcode"))

(defn ident 
  "
        Return identifier (PID) of process or `None` if it has yet to start
        "
  [ self ]
    (py/call-attr self "ident"))

(defn is-alive 
  "
        Return whether process is alive
        "
  [ self  ]
  (py/call-attr self "is_alive"  self  ))

(defn join 
  "
        Wait until child process terminates
        "
  [ self timeout ]
  (py/call-attr self "join"  self timeout ))

(defn kill 
  "
        Terminate process; sends SIGKILL signal or uses TerminateProcess()
        "
  [ self  ]
  (py/call-attr self "kill"  self  ))

(defn name 
  ""
  [ self ]
    (py/call-attr self "name"))

(defn pid 
  "
        Return identifier (PID) of process or `None` if it has yet to start
        "
  [ self ]
    (py/call-attr self "pid"))

(defn run 
  "
        Method to be run in sub-process; can be overridden in sub-class
        "
  [ self  ]
  (py/call-attr self "run"  self  ))

(defn sentinel 
  "
        Return a file descriptor (Unix) or handle (Windows) suitable for
        waiting for process termination.
        "
  [ self ]
    (py/call-attr self "sentinel"))

(defn start 
  "
        Start child process
        "
  [ self  ]
  (py/call-attr self "start"  self  ))

(defn terminate 
  "
        Terminate process; sends SIGTERM signal or uses TerminateProcess()
        "
  [ self  ]
  (py/call-attr self "terminate"  self  ))
