"""Loopback HTTP transport for an existing supervisor's public device API.

World ownership, serialization, journaling and request deduplication stay in the
supervisor. Owner lifecycle commands remain on its local Unix socket.
"""

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from roboenv.client import PUBLIC_OPS
from roboenv.validation import json_loads

from .protocol import ProtocolError, Request, encode_ndjson


def bind_http(supervisor, port):
    class Handler(BaseHTTPRequestHandler):
        def setup(self):
            super().setup()
            self.connection.settimeout(10)

        def log_message(self, *_args):
            pass

        def valid_endpoint(self):
            if self.path != "/":
                self.send_error(404, "Unknown endpoint")
                return False
            if self.headers.get("Origin") is not None:
                self.send_error(403, "Browser cross-origin requests are not supported")
                return False
            # Numeric loopback Host prevents DNS rebinding of this local service.
            expected_host = f"127.0.0.1:{self.server.server_port}"
            if self.headers.get("Host") != expected_host:
                self.send_error(403, "Use the advertised loopback endpoint")
                return False
            return True

        def do_GET(self):
            if not self.valid_endpoint():
                return
            data = encode_ndjson(
                {
                    "protocol": "roboenv/1",
                    "session": supervisor.config.session,
                    "generation": supervisor.config.generation,
                }
            )
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_POST(self):
            if not self.valid_endpoint():
                return
            if self.headers.get("Transfer-Encoding") is not None:
                self.send_error(400, "Use Content-Length")
                return
            lengths = self.headers.get_all("Content-Length", [])
            try:
                if len(lengths) != 1:
                    raise ValueError("Expected one content length")
                length = int(lengths[0])
                if not 0 < length <= supervisor.config.max_ndjson_bytes:
                    raise ValueError("Request size out of range")
            except ValueError:
                self.send_error(413, "Invalid request size")
                return
            if self.headers.get_content_type() != "application/json":
                self.send_error(415, "Use application/json")
                return
            raw = {}
            try:
                payload = self.rfile.read(length)
                if len(payload) != length:
                    raise ValueError("Incomplete request")
                raw = json_loads(payload)
                request = Request.from_mapping(raw, allowed_ops=PUBLIC_OPS)
                response = supervisor.handle(request)
            except (ValueError, ProtocolError) as exc:
                error = (
                    exc
                    if isinstance(exc, ProtocolError)
                    else ProtocolError("invalid_args", str(exc))
                )
                response = supervisor._unparsed_error(raw, error)
            except (TimeoutError, ConnectionError):
                return
            except Exception as exc:
                response = supervisor._unparsed_error(
                    raw, ProtocolError("internal_error", f"Request failed: {type(exc).__name__}")
                )
            data = encode_ndjson(response, max_bytes=supervisor.config.max_ndjson_bytes)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Connection", "close")
            self.end_headers()
            try:
                self.wfile.write(data)
            except (BrokenPipeError, ConnectionResetError):
                pass

    return ThreadingHTTPServer(("127.0.0.1", port), Handler)
