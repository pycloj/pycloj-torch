(ns torch.cuda.Popen
  " Execute a child program in a new process.

    For a complete description of the arguments see the Python documentation.

    Arguments:
      args: A string, or a sequence of program arguments.

      bufsize: supplied as the buffering argument to the open() function when
          creating the stdin/stdout/stderr pipe file objects

      executable: A replacement program to execute.

      stdin, stdout and stderr: These specify the executed programs' standard
          input, standard output and standard error file handles, respectively.

      preexec_fn: (POSIX only) An object to be called in the child process
          just before the child is executed.

      close_fds: Controls closing or inheriting of file descriptors.

      shell: If true, the command will be executed through the shell.

      cwd: Sets the current directory before the child is executed.

      env: Defines the environment variables for the new process.

      text: If true, decode stdin, stdout and stderr using the given encoding
          (if set) or the system default otherwise.

      universal_newlines: Alias of text, provided for backwards compatibility.

      startupinfo and creationflags (Windows only)

      restore_signals (POSIX only)

      start_new_session (POSIX only)

      pass_fds (POSIX only)

      encoding and errors: Text mode encoding and error handling to use for
          file objects stdin, stdout and stderr.

    Attributes:
        stdin, stdout, stderr, pid, returncode
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

(defn Popen 
  " Execute a child program in a new process.

    For a complete description of the arguments see the Python documentation.

    Arguments:
      args: A string, or a sequence of program arguments.

      bufsize: supplied as the buffering argument to the open() function when
          creating the stdin/stdout/stderr pipe file objects

      executable: A replacement program to execute.

      stdin, stdout and stderr: These specify the executed programs' standard
          input, standard output and standard error file handles, respectively.

      preexec_fn: (POSIX only) An object to be called in the child process
          just before the child is executed.

      close_fds: Controls closing or inheriting of file descriptors.

      shell: If true, the command will be executed through the shell.

      cwd: Sets the current directory before the child is executed.

      env: Defines the environment variables for the new process.

      text: If true, decode stdin, stdout and stderr using the given encoding
          (if set) or the system default otherwise.

      universal_newlines: Alias of text, provided for backwards compatibility.

      startupinfo and creationflags (Windows only)

      restore_signals (POSIX only)

      start_new_session (POSIX only)

      pass_fds (POSIX only)

      encoding and errors: Text mode encoding and error handling to use for
          file objects stdin, stdout and stderr.

    Attributes:
        stdin, stdout, stderr, pid, returncode
    "
  [args & {:keys [bufsize executable stdin stdout stderr preexec_fn close_fds shell cwd env universal_newlines startupinfo creationflags restore_signals start_new_session pass_fds encoding errors text]
                       :or {bufsize -1 close_fds true shell false creationflags 0 restore_signals true start_new_session false pass_fds ()}} ]
    (py/call-attr-kw cuda "Popen" [args] {:bufsize bufsize :executable executable :stdin stdin :stdout stdout :stderr stderr :preexec_fn preexec_fn :close_fds close_fds :shell shell :cwd cwd :env env :universal_newlines universal_newlines :startupinfo startupinfo :creationflags creationflags :restore_signals restore_signals :start_new_session start_new_session :pass_fds pass_fds :encoding encoding :errors errors :text text }))

(defn communicate 
  "Interact with process: Send data to stdin and close it.
        Read data from stdout and stderr, until end-of-file is
        reached.  Wait for process to terminate.

        The optional \"input\" argument should be data to be sent to the
        child process, or None, if no data should be sent to the child.
        communicate() returns a tuple (stdout, stderr).

        By default, all communication is in bytes, and therefore any
        \"input\" should be bytes, and the (stdout, stderr) will be bytes.
        If in text mode (indicated by self.text_mode), any \"input\" should
        be a string, and (stdout, stderr) will be strings decoded
        according to locale encoding, or by \"encoding\" if set. Text mode
        is triggered by setting any of text, encoding, errors or
        universal_newlines.
        "
  [ self input timeout ]
  (py/call-attr self "communicate"  self input timeout ))

(defn kill 
  "Kill the process with SIGKILL
            "
  [ self  ]
  (py/call-attr self "kill"  self  ))

(defn poll 
  "Check if child process has terminated. Set and return returncode
        attribute."
  [ self  ]
  (py/call-attr self "poll"  self  ))

(defn send-signal 
  "Send a signal to the process."
  [ self sig ]
  (py/call-attr self "send_signal"  self sig ))

(defn terminate 
  "Terminate the process with SIGTERM
            "
  [ self  ]
  (py/call-attr self "terminate"  self  ))

(defn universal-newlines 
  ""
  [ self ]
    (py/call-attr self "universal_newlines"))

(defn wait 
  "Wait for child process to terminate; returns self.returncode."
  [ self timeout ]
  (py/call-attr self "wait"  self timeout ))
