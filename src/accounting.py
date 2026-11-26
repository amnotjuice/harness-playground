"""Local proxy that bills every model request (Direct, selector and every OpenCode call).

Each request is forwarded unchanged; its cost is computed from the provider's own
token usage and the prices in src/opencode.json. Once an arm's recorded cost reaches
the target, further requests are refused, so the last request may overshoot.
"""
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import socket
import threading
import urllib.error
import urllib.request


def write(path, value):
    temporary = path.with_suffix(path.suffix + '.partial')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def cost_from_usage(usage, prices):
    incoming = usage.get('input_tokens', usage.get('prompt_tokens'))
    outgoing = usage.get('output_tokens', usage.get('completion_tokens'))
    details = usage.get('input_tokens_details', usage.get('prompt_tokens_details')) or {}
    cached = details.get('cached_tokens', 0) or 0
    written = details.get('cache_creation_tokens', details.get('cache_write_tokens', 0)) or 0
    for value in (incoming, outgoing, cached, written):
        if not isinstance(value, int) or value < 0:
            raise ValueError('Missing or invalid token usage')
    if cached + written > incoming:
        raise ValueError('Inconsistent cached token usage')
    # Published long-context rates: inputs above 272k tokens cost 2x input, 1.5x output.
    factor, output_factor = (2, 1.5) if incoming > 272000 else (1, 1)
    cost = ((incoming - cached - written) * prices['input'] * factor
            + cached * prices['cache_read'] * factor + written * prices['cache_write'] * factor
            + outgoing * prices['output'] * output_factor) / 1e6
    return dict(estimated_cost_usd=cost, observed_cost_usd=cost, model_invocations=1,
                input_tokens=incoming, output_tokens=outgoing, raw_usage=usage)


def final_response(data, content_type):
    """The response object carrying usage, from a JSON body or a server-sent event stream."""
    if 'event-stream' not in content_type:
        try:
            return json.loads(data)
        except ValueError:
            return None
    for line in reversed(data.decode(errors='replace').splitlines()):
        if line.startswith('data: '):
            try:
                event = json.loads(line[6:])
            except ValueError:
                continue
            if event.get('type') in ('response.completed', 'response.incomplete') and event.get('response', {}).get('usage'):
                return event['response']
    return None


# Provider errors with an error body are returned before generation and are not billed.
# Any other failure leaves the charge unknown, and no further request is sent.
NOT_BILLED_STATUS = {400, 401, 403, 404, 409, 413, 422, 429}


class Accounting:
    def __init__(self, output, backend, api_key, models, target=None):
        self.output = Path(output)
        self.backend = backend.rstrip('/')
        self.api_key = api_key
        self.models = models          # {'gpt-6.1-sol': {'input': ..., 'output': ..., ...}}
        self.target = target          # per-arm dollars; None means no target
        self.context = None
        self.lock = threading.RLock()
        (self.output / 'requests').mkdir(parents=True, exist_ok=True)

    def usages(self):
        return [json.loads(p.read_text()) for p in sorted((self.output / 'requests').glob('*/usage.json'))]

    def spent(self, arm=None):
        values = [v for v in self.usages() if arm is None or v['arm'] == arm]
        if any(v['estimated_cost_usd'] is None for v in values):
            return None
        return sum(v['estimated_cost_usd'] for v in values)

    def set_context(self, arm, directory):
        with self.lock:
            self.context = dict(arm=arm, directory=str(Path(directory).resolve()))

    def attempt_usage(self, directory):
        """Total recorded cost of every request made for one attempt directory."""
        with self.lock:
            values = [v for v in self.usages() if v['directory'] == str(Path(directory).resolve())]
        complete = all(v['estimated_cost_usd'] is not None for v in values)
        known = sum(v['observed_cost_usd'] for v in values)
        return dict(estimated_cost_usd=known if complete else None, observed_cost_usd=known,
                    model_invocations=len(values),
                    input_tokens=sum(v.get('input_tokens') or 0 for v in values),
                    output_tokens=sum(v.get('output_tokens') or 0 for v in values))

    def deny(self, reason, context=None):
        if context:
            write(Path(context['directory']) / 'request-stop.json', dict(reason=reason))
        return 402, 'application/json', json.dumps(
            {'error': {'message': 'HARNESS_STOP_' + reason, 'type': 'budget_error'}}).encode()

    def forward(self, path, raw):
        if path not in ('/v1/responses', '/v1/chat/completions'):
            return self.deny('unsupported_endpoint')
        with self.lock:
            # One request at a time, so every admission sees the latest recorded cost.
            if self.context is None:
                return self.deny('missing_request_context')
            context = dict(self.context)
            try:
                prices = self.models[json.loads(raw)['model'].removeprefix('openai/')]
            except (ValueError, KeyError):
                return self.deny('unpriced_model', context)
            arm_spent = self.spent(context['arm'])
            if arm_spent is None:
                return self.deny('unknown_billing', context)
            if self.target is not None and arm_spent >= self.target:
                return self.deny('target', context)
            folder = self.output / 'requests' / f'{len(self.usages()) + 1:05d}'
            folder.mkdir()
            (folder / 'request.json').write_bytes(raw)
            write(folder / 'usage.json', dict(context, estimated_cost_usd=None, observed_cost_usd=0.))
            request = urllib.request.Request(self.backend + path.removeprefix('/v1'), data=raw, method='POST',
                                             headers={'Authorization': 'Bearer ' + self.api_key,
                                                      'Content-Type': 'application/json'})
            try:
                with urllib.request.urlopen(request) as response:
                    content_type = response.headers.get('Content-Type', 'application/json')
                    data = response.read()
                (folder / 'response.raw').write_bytes(data)
                final = final_response(data, content_type)
                if not final or final.get('service_tier') not in (None, 'auto', 'default', 'standard'):
                    raise ValueError('Missing usage or unpriced service tier')
                write(folder / 'usage.json', dict(context, **cost_from_usage(final['usage'], prices)))
                return 200, content_type, data
            except urllib.error.HTTPError as exc:
                body = exc.read()
                (folder / 'response.raw').write_bytes(body)
                try:
                    payload = json.loads(body)
                except ValueError:
                    payload = None
                if exc.code in NOT_BILLED_STATUS and isinstance(payload, dict) and isinstance(payload.get('error'), dict):
                    write(folder / 'usage.json', dict(context, estimated_cost_usd=0., observed_cost_usd=0.,
                                                      model_invocations=1, note=f'HTTP {exc.code}; not billed'))
                    return exc.code, exc.headers.get('Content-Type', 'application/json'), body
                write(folder / 'failure.json', dict(error=f'HTTP {exc.code}', billing='unknown'))
            except urllib.error.URLError as exc:
                if isinstance(exc.reason, (socket.gaierror, ConnectionRefusedError)):
                    # Nothing reached the provider.
                    write(folder / 'usage.json', dict(context, estimated_cost_usd=0., observed_cost_usd=0.,
                                                      model_invocations=1, note=f'not sent: {exc.reason}'))
                    return self.deny('not_sent', context)
                write(folder / 'failure.json', dict(error=f'{type(exc).__name__}: {exc}', billing='unknown'))
            except Exception as exc:
                write(folder / 'failure.json', dict(error=f'{type(exc).__name__}: {exc}', billing='unknown'))
            return 502, 'application/json', json.dumps(
                {'error': {'message': 'HARNESS_UNKNOWN_BILLING', 'type': 'api_error'}}).encode()

    @contextmanager
    def serve(self):
        accounting = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                status, kind, data = accounting.forward(self.path, self.rfile.read(int(self.headers['Content-Length'])))
                self.send_response(status)
                self.send_header('Content-Type', kind)
                self.send_header('Content-Length', str(len(data)))
                self.end_headers()
                try:
                    self.wfile.write(data)
                except (BrokenPipeError, ConnectionResetError):
                    pass

        server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            yield f'http://127.0.0.1:{server.server_port}/v1'
        finally:
            server.shutdown()
            server.server_close()
