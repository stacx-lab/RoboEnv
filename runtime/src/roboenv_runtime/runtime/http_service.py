"""Loopback HTTP transport for an existing supervisor's public device API.

World ownership, serialization, journaling and request deduplication stay in the
supervisor. Owner lifecycle commands remain on its local Unix socket.
"""

import re
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from roboenv.devices.contract import SERVICE_OPS

PUBLIC_OPS = SERVICE_OPS | {"health", "simulation.status", "simulation.stop"}
from roboenv.validation import json_loads

from .protocol import ProtocolError, Request, encode_ndjson


def bind_http(supervisor, port, host="127.0.0.1"):
    class Server(ThreadingHTTPServer):
        def __init__(self, *args):
            self.connections = set()
            self.connections_lock = threading.Lock()
            super().__init__(*args)

        def get_request(self):
            connection, address = super().get_request()
            with self.connections_lock:
                self.connections.add(connection)
            return connection, address

        def shutdown_request(self, connection):
            with self.connections_lock:
                self.connections.discard(connection)
            super().shutdown_request(connection)

        def server_close(self):
            with self.connections_lock:
                for connection in self.connections:
                    try:
                        connection.shutdown(socket.SHUT_RDWR)
                    except OSError:
                        pass
                    connection.close()
                self.connections.clear()
            super().server_close()

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        def setup(self):
            super().setup()
            self.connection.settimeout(10)
            self.connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

        def log_message(self, *_args):
            pass

        def valid_endpoint(self):
            if self.path != "/" and not (self.command == "GET" and re.fullmatch(r"/artifacts/[A-Za-z0-9_.-]+/[0-9]+/[a-f0-9]{32}\.(?:png|pgm)", self.path)):
                self.send_error(404, "Unknown endpoint")
                return False
            if self.headers.get("Origin") is not None:
                self.send_error(403, "Browser cross-origin requests are not supported")
                return False
            # Numeric loopback Host prevents DNS rebinding of this local service.
            expected_host = f"127.0.0.1:{self.server.server_port}"
            if host == "127.0.0.1" and self.headers.get("Host") != expected_host:
                self.send_error(403, "Use the advertised loopback endpoint")
                return False
            return True

        def do_GET(self):
            if (self.headers.get("Transfer-Encoding") is not None
                    or self.headers.get_all("Content-Length", ["0"]) != ["0"]):
                self.send_error(400, "GET requests must not have a body")
                return
            if not self.valid_endpoint():
                return
            if self.path.startswith("/artifacts/"):
                _, _, session, generation, name = self.path.split("/")
                if session != supervisor.config.session or int(generation) != supervisor.config.generation:
                    self.send_error(409, "Episode changed")
                    return
                file = Path(supervisor.config.run_dir) / "captures" / name
                if file.is_symlink() or not file.is_file():
                    self.send_error(404, "Unknown artifact")
                    return
                data = file.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "image/png" if file.suffix == ".png" else "image/x-portable-graymap")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
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
                    self.close_connection = True
                    raise ValueError("Incomplete request")
                raw = json_loads(payload)
                request = Request.from_mapping(raw, allowed_ops=PUBLIC_OPS)
                response = supervisor.handle(request, public=True)
            except (ValueError, ProtocolError) as exc:
                error = (
                    exc
                    if isinstance(exc, ProtocolError)
                    else ProtocolError("invalid_args", str(exc))
                )
                response = supervisor._unparsed_error(raw, error)
            except (TimeoutError, ConnectionError):
                self.close_connection = True
                return
            except Exception as exc:
                response = supervisor._unparsed_error(
                    raw, ProtocolError("internal_error", f"Request failed: {type(exc).__name__}")
                )
            data = encode_ndjson(response, max_bytes=supervisor.config.max_ndjson_bytes)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            try:
                self.wfile.write(data)
            except (BrokenPipeError, ConnectionResetError):
                self.close_connection = True

    return Server((host, port), Handler)
