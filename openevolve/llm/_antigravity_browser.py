"""Browser helper: keep browser subprocess output out of the ACP JSON stream."""

import os
import sys
import webbrowser
from urllib.parse import urlparse

if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] not in ("--open", "--block"):
        raise SystemExit(1)
    if sys.argv[1] == "--block":
        print("OPENEVOLVE_ACP_AUTH_REQUIRED", file=sys.stderr, flush=True)
        raise SystemExit(0)
    if urlparse(sys.argv[2]).hostname != "accounts.google.com":
        raise SystemExit(1)
    # The helper inherits the ACP pipes; redirect the underlying file descriptors
    # so browser startup messages cannot become malformed JSON-RPC responses.
    if os.name == "posix":
        os.setsid()
    with open(os.devnull, "wb") as sink:
        os.dup2(sink.fileno(), 1)
        os.dup2(sink.fileno(), 2)
    os.environ.pop("BROWSER", None)
    webbrowser.open(sys.argv[2], new=1)
