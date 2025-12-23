from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler

class Handler(SimpleHTTPRequestHandler):
    # Явно задаём MIME-типы для модулей/wasm
    extensions_map = SimpleHTTPRequestHandler.extensions_map.copy()
    extensions_map.update({
        ".mjs": "application/javascript",
        ".js": "application/javascript",
        ".wasm": "application/wasm",
    })

if __name__ == "__main__":
    ThreadingHTTPServer(("localhost", 8000), Handler).serve_forever()
