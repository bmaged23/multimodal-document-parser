"""Serve only a prepared source-reference workspace on localhost; persist review revisions."""

import argparse
import json
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit

from multimodal_doc_parser.evaluation.reference_review import (
    ReviewStore,
    ReviewUpdate,
    RevisionConflict,
)


class ReviewHandler(BaseHTTPRequestHandler):
    def __init__(self, *args, store, **kwargs):
        self.store = store
        super().__init__(*args, **kwargs)

    def reply(self, status, value, mime="application/json"):
        body = (
            json.dumps(value, ensure_ascii=False).encode() if mime == "application/json" else value
        )
        self.send_response(status)
        self.send_header("Content-Type", mime + ("; charset=utf-8" if mime != "image/png" else ""))
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = unquote(urlsplit(self.path).path)
        try:
            if path == "/":
                import multimodal_doc_parser.evaluation.reference_review as module

                self.reply(
                    200, Path(module.__file__).with_suffix(".html").read_bytes(), "text/html"
                )
            elif path == "/api/dataset":
                self.reply(200, self.store.snapshot())
            elif path == "/api/export":
                self.reply(200, self.store.export())
            elif path.startswith("/source/"):
                _, _, case, number = path.split("/")
                self.reply(200, self.store.source(case, int(number)).read_bytes(), "image/png")
            else:
                self.reply(404, dict(error="Not found"))
        except (KeyError, ValueError, OSError):
            self.reply(400, dict(error="Unknown source or changed reference workspace"))

    def do_POST(self):
        origin = self.headers.get("Origin")
        if self.headers.get("X-Reference-Review") != "1" or (
            origin and origin != "http://" + self.headers.get("Host", "")
        ):
            self.reply(403, dict(error="Review requests must come from this local interface"))
            return
        path = unquote(urlsplit(self.path).path)
        if not path.startswith("/api/annotations/") or len(path.split("/")) != 4:
            self.reply(404, dict(error="Not found"))
            return
        try:
            size = int(self.headers.get("Content-Length", "0"))
            if not 0 < size <= 2 * 1024**2:
                raise ValueError("Invalid request size")
            update = ReviewUpdate.model_validate_json(self.rfile.read(size))
            self.reply(200, self.store.update(path.split("/")[-1], update))
        except RevisionConflict as error:
            self.reply(409, dict(error=str(error)))
        except (ValueError, KeyError) as error:
            self.reply(400, dict(error=str(error)))

    def log_message(self, format, *args):
        pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--port", type=int, default=8766)
    args = parser.parse_args()
    store = ReviewStore(args.directory)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), partial(ReviewHandler, store=store))
    print(f"Source reference review: http://localhost:{args.port}/", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
