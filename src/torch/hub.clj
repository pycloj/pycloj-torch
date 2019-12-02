(ns torch.hub
  ""
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce hub (import-module "torch.hub"))

(defn download-url-to-file 
  "Download object at the given URL to a local path.

    Args:
        url (string): URL of the object to download
        dst (string): Full path where object will be saved, e.g. `/tmp/temporary_file`
        hash_prefix (string, optional): If not None, the SHA256 downloaded file should start with `hash_prefix`.
            Default: None
        progress (bool, optional): whether or not to display a progress bar to stderr
            Default: True

    Example:
        >>> torch.hub.download_url_to_file('https://s3.amazonaws.com/pytorch/models/resnet18-5c106cde.pth', '/tmp/temporary_file')

    "
  [url dst hash_prefix & {:keys [progress]
                       :or {progress true}} ]
    (py/call-attr-kw hub "download_url_to_file" [url dst hash_prefix] {:progress progress }))

(defn help 
  "
    Show the docstring of entrypoint `model`.

    Args:
        github (string): a string with format <repo_owner/repo_name[:tag_name]> with an optional
            tag/branch. The default branch is `master` if not specified.
            Example: 'pytorch/vision[:hub]'
        model (string): a string of entrypoint name defined in repo's hubconf.py
        force_reload (bool, optional): whether to discard the existing cache and force a fresh download.
            Default is `False`.
    Example:
        >>> print(torch.hub.help('pytorch/vision', 'resnet18', force_reload=True))
    "
  [github model & {:keys [force_reload]
                       :or {force_reload false}} ]
    (py/call-attr-kw hub "help" [github model] {:force_reload force_reload }))

(defn import-module 
  ""
  [ name path ]
  (py/call-attr hub "import_module"  name path ))

(defn list 
  "
    List all entrypoints available in `github` hubconf.

    Args:
        github (string): a string with format \"repo_owner/repo_name[:tag_name]\" with an optional
            tag/branch. The default branch is `master` if not specified.
            Example: 'pytorch/vision[:hub]'
        force_reload (bool, optional): whether to discard the existing cache and force a fresh download.
            Default is `False`.
    Returns:
        entrypoints: a list of available entrypoint names

    Example:
        >>> entrypoints = torch.hub.list('pytorch/vision', force_reload=True)
    "
  [github & {:keys [force_reload]
                       :or {force_reload false}} ]
    (py/call-attr-kw hub "list" [github] {:force_reload force_reload }))

(defn load 
  "
    Load a model from a github repo, with pretrained weights.

    Args:
        github (string): a string with format \"repo_owner/repo_name[:tag_name]\" with an optional
            tag/branch. The default branch is `master` if not specified.
            Example: 'pytorch/vision[:hub]'
        model (string): a string of entrypoint name defined in repo's hubconf.py
        *args (optional): the corresponding args for callable `model`.
        force_reload (bool, optional): whether to force a fresh download of github repo unconditionally.
            Default is `False`.
        verbose (bool, optional): If False, mute messages about hitting local caches. Note that the message
            about first download is cannot be muted.
            Default is `True`.
        **kwargs (optional): the corresponding kwargs for callable `model`.

    Returns:
        a single model with corresponding pretrained weights.

    Example:
        >>> model = torch.hub.load('pytorch/vision', 'resnet50', pretrained=True)
    "
  [ github model ]
  (py/call-attr hub "load"  github model ))

(defn load-state-dict-from-url 
  "Loads the Torch serialized object at the given URL.

    If downloaded file is a zip file, it will be automatically
    decompressed.

    If the object is already present in `model_dir`, it's deserialized and
    returned.
    The default value of `model_dir` is ``$TORCH_HOME/checkpoints`` where
    environment variable ``$TORCH_HOME`` defaults to ``$XDG_CACHE_HOME/torch``.
    ``$XDG_CACHE_HOME`` follows the X Design Group specification of the Linux
    filesytem layout, with a default value ``~/.cache`` if not set.

    Args:
        url (string): URL of the object to download
        model_dir (string, optional): directory in which to save the object
        map_location (optional): a function or a dict specifying how to remap storage locations (see torch.load)
        progress (bool, optional): whether or not to display a progress bar to stderr.
            Default: True
        check_hash(bool, optional): If True, the filename part of the URL should follow the naming convention
            ``filename-<sha256>.ext`` where ``<sha256>`` is the first eight or more
            digits of the SHA256 hash of the contents of the file. The hash is used to
            ensure unique names and to verify the contents of the file.
            Default: False

    Example:
        >>> state_dict = torch.hub.load_state_dict_from_url('https://s3.amazonaws.com/pytorch/models/resnet18-5c106cde.pth')

    "
  [url model_dir map_location & {:keys [progress check_hash]
                       :or {progress true check_hash false}} ]
    (py/call-attr-kw hub "load_state_dict_from_url" [url model_dir map_location] {:progress progress :check_hash check_hash }))

(defn set-dir 
  "
    Optionally set hub_dir to a local dir to save downloaded models & weights.

    If ``set_dir`` is not called, default path is ``$TORCH_HOME/hub`` where
    environment variable ``$TORCH_HOME`` defaults to ``$XDG_CACHE_HOME/torch``.
    ``$XDG_CACHE_HOME`` follows the X Design Group specification of the Linux
    filesytem layout, with a default value ``~/.cache`` if the environment
    variable is not set.


    Args:
        d (string): path to a local folder to save downloaded models & weights.
    "
  [ d ]
  (py/call-attr hub "set_dir"  d ))

(defn urlopen 
  "Open the URL url, which can be either a string or a Request object.

    *data* must be an object specifying additional data to be sent to
    the server, or None if no such data is needed.  See Request for
    details.

    urllib.request module uses HTTP/1.1 and includes a \"Connection:close\"
    header in its HTTP requests.

    The optional *timeout* parameter specifies a timeout in seconds for
    blocking operations like the connection attempt (if not specified, the
    global default timeout setting will be used). This only works for HTTP,
    HTTPS and FTP connections.

    If *context* is specified, it must be a ssl.SSLContext instance describing
    the various SSL options. See HTTPSConnection for more details.

    The optional *cafile* and *capath* parameters specify a set of trusted CA
    certificates for HTTPS requests. cafile should point to a single file
    containing a bundle of CA certificates, whereas capath should point to a
    directory of hashed certificate files. More information can be found in
    ssl.SSLContext.load_verify_locations().

    The *cadefault* parameter is ignored.

    This function always returns an object which can work as a context
    manager and has methods such as

    * geturl() - return the URL of the resource retrieved, commonly used to
      determine if a redirect was followed

    * info() - return the meta-information of the page, such as headers, in the
      form of an email.message_from_string() instance (see Quick Reference to
      HTTP Headers)

    * getcode() - return the HTTP status code of the response.  Raises URLError
      on errors.

    For HTTP and HTTPS URLs, this function returns a http.client.HTTPResponse
    object slightly modified. In addition to the three new methods above, the
    msg attribute contains the same information as the reason attribute ---
    the reason phrase returned by the server --- instead of the response
    headers as it is specified in the documentation for HTTPResponse.

    For FTP, file, and data URLs and requests explicitly handled by legacy
    URLopener and FancyURLopener classes, this function returns a
    urllib.response.addinfourl object.

    Note that None may be returned if no handler handles the request (though
    the default installed global OpenerDirector uses UnknownHandler to ensure
    this never happens).

    In addition, if proxy settings are detected (for example, when a *_proxy
    environment variable like http_proxy is set), ProxyHandler is default
    installed and makes sure the requests are handled through the proxy.

    "
  [url data & {:keys [timeout cafile capath cadefault context]
                       :or {timeout <object object at 0x10c4429e0> cadefault false}} ]
    (py/call-attr-kw hub "urlopen" [url data] {:timeout timeout :cafile cafile :capath capath :cadefault cadefault :context context }))

(defn urlparse 
  "Parse a URL into 6 components:
    <scheme>://<netloc>/<path>;<params>?<query>#<fragment>
    Return a 6-tuple: (scheme, netloc, path, params, query, fragment).
    Note that we don't break the components up in smaller bits
    (e.g. netloc is a single string) and we don't expand % escapes."
  [url & {:keys [scheme allow_fragments]
                       :or {scheme "" allow_fragments true}} ]
    (py/call-attr-kw hub "urlparse" [url] {:scheme scheme :allow_fragments allow_fragments }))
