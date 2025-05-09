#!/usr/bin/env python3
import http.server
import socketserver
import sys
import os
import time
from urllib.parse import urlparse

# Configuration
PORT = 3000
DIRECTORY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")

class CustomHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        # Set the directory to serve files from
        super().__init__(*args, directory=DIRECTORY, **kwargs)
    
    def do_GET(self):
        print(f"[{time.strftime('%H:%M:%S')}] GET request for {self.path}")
        return super().do_GET()
    
    def end_headers(self):
        # Add CORS headers
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        super().end_headers()
    
    def log_message(self, format, *args):
        # Custom logging
        sys.stderr.write(f"[{time.strftime('%H:%M:%S')}] {self.address_string()} - {format % args}\n")

def main():
    # Change to the script's directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    print(f"[{time.strftime('%H:%M:%S')}] Starting server in directory: {DIRECTORY}")
    print(f"[{time.strftime('%H:%M:%S')}] Server will run on http://127.0.0.1:{PORT}")

    # Create server with specific binding
    try:
        with socketserver.TCPServer(("127.0.0.1", PORT), CustomHTTPRequestHandler) as httpd:
            print(f"[{time.strftime('%H:%M:%S')}] Server started successfully!")
            print(f"[{time.strftime('%H:%M:%S')}] Open your browser to http://127.0.0.1:{PORT}/")
            print(f"[{time.strftime('%H:%M:%S')}] Press Ctrl+C to stop the server")
            httpd.serve_forever()
    except OSError as e:
        if e.errno == 48:  # Address already in use
            print(f"[{time.strftime('%H:%M:%S')}] ERROR: Port {PORT} is already in use!")
            print(f"[{time.strftime('%H:%M:%S')}] Try to kill any processes using port {PORT} with:")
            print(f"    lsof -i :{PORT} | grep LISTEN")
            print(f"    kill -9 [PID]")
        else:
            print(f"[{time.strftime('%H:%M:%S')}] Server error: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print(f"[{time.strftime('%H:%M:%S')}] Server stopped by user")
        sys.exit(0)

if __name__ == "__main__":
    main() 