from pathlib import Path
from functools import partial
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from urllib.parse import urlsplit
import argparse, subprocess, sys

class DemoHandler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Cache-Control', 'no-cache')
        super().end_headers()

    def do_GET(self):
        if self.server.landing and urlsplit(self.path).path == '/':
            self.send_response(302)
            self.send_header('Location', self.server.landing)
            self.end_headers()
            return
        super().do_GET()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Host site on :8766 (default) or the student-only site on :8767 (--student).')
    parser.add_argument('--student', action='store_true', help='serve dist-student/ (rebuilt first), with no host models or pages')
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    if args.student:
        subprocess.run([sys.executable, str(root / 'scripts/build_student_site.py')], check=True)
        directory, port, landing, name = root / 'dist-student', 8767, None, 'Puck Lab (student site)'
        url = f'http://127.0.0.1:{port}/'
    else:
        directory, port, landing, name = root, 8766, '/app/index.html?host=1', 'Puck Lab (host site)'
        url = f'http://127.0.0.1:{port}/'
    server = ThreadingHTTPServer(('127.0.0.1', port), partial(DemoHandler, directory=str(directory)))
    server.landing = landing
    print(f'{name}: {url}', flush=True)
    server.serve_forever()
