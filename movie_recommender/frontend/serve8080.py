#!/usr/bin/env python3
import http.server
import socketserver
import sys
import os
import time

# Configuration
PORT = 8080
DIRECTORY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")

class CustomHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        # Set the directory to serve files from
        super().__init__(*args, directory=DIRECTORY, **kwargs)
    
    def end_headers(self):
        # Add CORS headers
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        super().end_headers()

def main():
    print(f"Starting server in directory: {DIRECTORY}")
    print(f"Server will run on http://localhost:{PORT}")

    # Create server with specific binding
    try:
        with socketserver.TCPServer(("", PORT), CustomHTTPRequestHandler) as httpd:
            print(f"Server started successfully!")
            print(f"Open your browser to http://localhost:{PORT}/")
            print(f"Press Ctrl+C to stop the server")
            httpd.serve_forever()
    except OSError as e:
        print(f"Server error: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print(f"Server stopped by user")
        sys.exit(0)

if __name__ == "__main__":
    main() 