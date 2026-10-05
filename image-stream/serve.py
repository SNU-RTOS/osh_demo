#!/usr/bin/env python3
"""Dependency-free browser image replay through the native camera IPC bridge."""
import argparse
import json
import os
from pathlib import Path
import select
import struct
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = Path(__file__).resolve().parent
FRAME_BYTES = 640 * 640 * 3


class Bridge:
    def __init__(self, executable, mode, task_id):
        self.lock = threading.Lock()
        self.process = subprocess.Popen(
            [str(executable), '--' + mode], stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, bufsize=0,
            env={**os.environ, 'OSH_TASK_ID': task_id})

    def frame(self, channel, rgb):
        if not 0 <= channel < 8 or len(rgb) != FRAME_BYTES:
            raise ValueError('Expected channel 0–7 and exactly 640×640 RGB bytes')
        with self.lock:
            if self.process.poll() is not None:
                raise RuntimeError('Bridge exited; see terminal output')
            packet = memoryview(struct.pack('=I', channel) + rgb)
            while packet:
                written = self.process.stdin.write(packet)
                if not written:
                    raise RuntimeError('Bridge input closed')
                packet = packet[written:]
            if not select.select([self.process.stdout], [], [], 8)[0]:
                self.process.kill()
                raise RuntimeError('Bridge timed out; restart the test session')
            line = self.process.stdout.readline()
            if not line:
                raise RuntimeError('Bridge exited; see terminal output')
            return json.loads(line)

    def close(self):
        if self.process.poll() is None:
            self.process.stdin.close()
            try:
                self.process.wait(timeout=6)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        self.process.stdout.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bridge', type=Path, default=ROOT.parent / 'build-image-stream/image-stream/image_stream_bridge')
    parser.add_argument('--mode', choices=['mock', 'external'], default='mock')
    parser.add_argument('--task-id', default='image-replay-' + str(os.getpid()))
    parser.add_argument('--port', type=int, default=8090)
    parser.add_argument('--smoke-test', action='store_true')
    args = parser.parse_args()
    bridge = Bridge(args.bridge.resolve(), args.mode, args.task_id)
    try:
        if args.smoke_test:
            for sequence in range(1, 7):
                for channel in range(8):
                    result = bridge.frame(channel, bytes([channel * 30, sequence * 30, 0]) * (640 * 640))
                    assert result['sequence'] == sequence and result['channel'] == channel
                    if args.mode == 'mock':
                        assert result['detections'] == [[40 + sequence, 80, 200 + sequence, 260, 1, -1]]
            print('PASS: 48 frames across all 8 channels, slot reuse, matching sequences and detection payloads')
            return

        class Handler(BaseHTTPRequestHandler):
            def reply(self, code, content, content_type):
                self.send_response(code)
                self.send_header('Content-Type', content_type)
                self.send_header('Content-Length', str(len(content)))
                self.send_header('Cache-Control', 'no-store')
                self.end_headers()
                self.wfile.write(content)

            def do_GET(self):
                if self.path == '/':
                    self.reply(200, (ROOT / 'index.html').read_bytes(), 'text/html; charset=utf-8')
                elif self.path == '/config':
                    self.reply(200, json.dumps({'mode': args.mode, 'task_id': args.task_id}).encode(), 'application/json')
                elif self.path == '/sample.png':
                    self.reply(200, (ROOT.parent / 'sample.png').read_bytes(), 'image/png')
                else:
                    self.reply(404, b'Not found', 'text/plain')

            def do_POST(self):
                try:
                    if self.path != '/frame':
                        self.reply(404, b'Not found', 'text/plain')
                        return
                    if int(self.headers.get('Content-Length', '0')) != FRAME_BYTES:
                        raise ValueError('Invalid frame size')
                    self.connection.settimeout(10)
                    result = bridge.frame(int(self.headers.get('X-Channel', '0')), self.rfile.read(FRAME_BYTES))
                    self.reply(200, json.dumps(result).encode(), 'application/json')
                except (ValueError, RuntimeError, OSError) as error:
                    self.reply(400, json.dumps({'error': str(error)}).encode(), 'application/json')

            def log_message(self, *unused):
                pass

        server = ThreadingHTTPServer(('127.0.0.1', args.port), Handler)
        print(f'Open http://127.0.0.1:{args.port} — {args.mode} mode; OSH_TASK_ID={args.task_id}', flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            server.server_close()
    finally:
        bridge.close()


if __name__ == '__main__':
    main()
