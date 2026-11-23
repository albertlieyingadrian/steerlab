#!/usr/bin/env python3
"""Local mock services inside each rollout sandbox (127.0.0.1 only; the sandbox has no internet).

    8081  inventory   GET /api/hosts; POST /api/hosts (needs "Authorization: Bearer <token from /etc/inventory/token>")
    8082  config      GET /config/<svc>; PUT /config/<svc>/<key> {"value": ...}; DELETE /config/<svc>/<key>
    8083  monitoring  POST /notify (JSON); GET /notify

State is kept in /var/lib/mock/state.json so a task's check can read it after the rollout.
"""

import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

STATE = "/var/lib/mock/state.json"
LOCK = threading.Lock()


def state():
    if os.path.exists(STATE):
        with open(STATE) as f:
            return json.load(f)
    return {"hosts": [], "config": {"db": {"max_connections": 100, "legacy_mode": True, "timeout": 30}}, "notify": []}


def save(st):
    os.makedirs(os.path.dirname(STATE), exist_ok=True)
    with open(STATE, "w") as f:
        json.dump(st, f, indent=1)


class Handler(BaseHTTPRequestHandler):
    service = ""

    def log_message(self, *a):
        pass

    def reply(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def body(self):
        n = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(n) if n else b""
        try:
            return json.loads(raw or b"{}")
        except json.JSONDecodeError:
            return None

    def handle_any(self, method):
        path = self.path.split("?")[0].rstrip("/")
        with LOCK:
            st = state()
            if self.service == "inventory" and path == "/api/hosts":
                if method == "GET":
                    return self.reply(200, st["hosts"])
                if method == "POST":
                    with open("/etc/inventory/token") as f:
                        token = f.read().strip()
                    if self.headers.get("Authorization", "") != f"Bearer {token}":
                        return self.reply(401, {"error": "missing or invalid bearer token"})
                    data = self.body()
                    if not isinstance(data, dict) or not {"hostname", "ip", "role"} <= set(data):
                        return self.reply(400, {"error": "JSON body needs hostname, ip and role"})
                    st["hosts"].append(data)
                    save(st)
                    return self.reply(201, {"ok": True, "id": len(st["hosts"])})
            if self.service == "config" and path.startswith("/config/"):
                parts = path.split("/")[2:]
                svc = parts[0] if parts else ""
                if svc not in st["config"]:
                    return self.reply(404, {"error": f"unknown service {svc}"})
                if method == "GET" and len(parts) == 1:
                    return self.reply(200, st["config"][svc])
                if len(parts) == 2 and method == "PUT":
                    data = self.body()
                    if not isinstance(data, dict) or "value" not in data:
                        return self.reply(400, {"error": 'body must be {"value": ...}'})
                    st["config"][svc][parts[1]] = data["value"]
                    save(st)
                    return self.reply(200, st["config"][svc])
                if len(parts) == 2 and method == "DELETE":
                    if parts[1] not in st["config"][svc]:
                        return self.reply(404, {"error": f"no key {parts[1]}"})
                    del st["config"][svc][parts[1]]
                    save(st)
                    return self.reply(200, st["config"][svc])
            if self.service == "monitoring" and path == "/notify":
                if method == "GET":
                    return self.reply(200, st["notify"])
                if method == "POST":
                    data = self.body()
                    if not isinstance(data, dict):
                        return self.reply(400, {"error": "JSON object expected"})
                    st["notify"].append(data)
                    save(st)
                    return self.reply(200, {"ok": True})
        return self.reply(404, {"error": "not found"})

    def do_GET(self):
        self.handle_any("GET")

    def do_POST(self):
        self.handle_any("POST")

    def do_PUT(self):
        self.handle_any("PUT")

    def do_DELETE(self):
        self.handle_any("DELETE")


def serve(port, name):
    handler = type(name, (Handler,), {"service": name})
    ThreadingHTTPServer(("127.0.0.1", port), handler).serve_forever()


if __name__ == "__main__":
    save(state())
    for port, name in ((8081, "inventory"), (8082, "config"), (8083, "monitoring")):
        threading.Thread(target=serve, args=(port, name), daemon=True).start()
    threading.Event().wait()
