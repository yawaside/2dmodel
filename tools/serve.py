#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Dev server for the browser preview.

`python3 -m http.server` lets the browser cache the model, and a freshly built
.moc3 then gets rendered with the *previous* texture - which looks like the
model has completely fallen apart.  So everything is served with `no-store`.

    python3 tools/serve.py [root] [port]
"""

from __future__ import annotations

import http.server
import os
import sys


class Handler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        super().end_headers()

    def log_message(self, fmt, *args):
        pass


def main() -> None:
    root = sys.argv[1] if len(sys.argv) > 1 else "."
    port = int(sys.argv[2]) if len(sys.argv) > 2 else 8000
    os.chdir(root)
    srv = http.server.ThreadingHTTPServer(("0.0.0.0", port), Handler)
    print("serving %s on http://0.0.0.0:%d" % (os.getcwd(), port))
    srv.serve_forever()


if __name__ == "__main__":
    main()
