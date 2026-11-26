import json
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from src.accounting import Accounting, cost_from_usage

PRICES = {'input': 2, 'output': 10, 'cache_read': .1, 'cache_write': 2.5}


class FakeProvider:
    """Answers each request with the next queued (status, body, content type)."""
    def __init__(self, replies):
        self.replies = list(replies)
        self.received = []
        provider = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                provider.received.append(self.rfile.read(int(self.headers['Content-Length'])))
                status, body, kind = provider.replies.pop(0)
                self.send_response(status)
                self.send_header('Content-Type', kind)
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        self.server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.url = f'http://127.0.0.1:{self.server.server_port}/v1'


def chat(tokens_in, tokens_out):
    return 200, json.dumps({'usage': {'prompt_tokens': tokens_in, 'completion_tokens': tokens_out}}).encode(), 'application/json'


def post(proxy, path='/chat/completions', model='openai/gpt-6.1-sol'):
    request = urllib.request.Request(proxy + path, data=json.dumps({'model': model}).encode(), method='POST',
                                     headers={'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(request) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()


class AccountingTests(unittest.TestCase):
    def test_cost_uses_cached_and_long_context_rates(self):
        self.assertAlmostEqual(cost_from_usage({'input_tokens': 1000, 'output_tokens': 100,
                                                'input_tokens_details': {'cached_tokens': 400}}, PRICES)['estimated_cost_usd'],
                               (600 * 2 + 400 * .1 + 100 * 10) / 1e6)
        self.assertAlmostEqual(cost_from_usage({'input_tokens': 300000, 'output_tokens': 10}, PRICES)['estimated_cost_usd'],
                               (300000 * 2 * 2 + 10 * 10 * 1.5) / 1e6)

    def test_forwards_unchanged_bills_each_request_and_refuses_after_target(self):
        provider = FakeProvider([chat(100000, 10000), chat(100000, 10000)])
        with tempfile.TemporaryDirectory() as temp:
            accounting = Accounting(Path(temp), provider.url, 'real-key', {'gpt-6.1-sol': PRICES}, target=.25)
            attempt = Path(temp) / 'attempt'
            attempt.mkdir()
            accounting.set_context('direct', attempt)
            with accounting.serve() as proxy:
                self.assertEqual(post(proxy)[0], 200)   # $0.30 crosses the $0.25 target
                status, body = post(proxy)
            self.assertEqual(status, 402)
            self.assertIn(b'HARNESS_STOP_target', body)
            self.assertEqual(json.loads((attempt / 'request-stop.json').read_text())['reason'], 'target')
            self.assertEqual(provider.received, [json.dumps({'model': 'openai/gpt-6.1-sol'}).encode()])
            usage = accounting.attempt_usage(attempt)
            self.assertAlmostEqual(usage['estimated_cost_usd'], .3)
            self.assertEqual(usage['model_invocations'], 1)
            self.assertAlmostEqual(accounting.spent('opencode'), 0)

    def test_streamed_response_usage_is_billed(self):
        events = [{'type': 'response.created', 'response': {'id': 'r'}},
                  {'type': 'response.completed', 'response': {'id': 'r', 'usage': {'input_tokens': 10, 'output_tokens': 5}}}]
        body = ''.join('data: ' + json.dumps(e) + '\n\n' for e in events).encode()
        provider = FakeProvider([(200, body, 'text/event-stream')])
        with tempfile.TemporaryDirectory() as temp:
            accounting = Accounting(Path(temp), provider.url, 'k', {'gpt-6.1-sol': PRICES})
            accounting.set_context('opencode', Path(temp))
            with accounting.serve() as proxy:
                self.assertEqual(post(proxy, '/responses'), (200, body))
            self.assertAlmostEqual(accounting.spent(), (10 * 2 + 5 * 10) / 1e6)

    def test_rejected_requests_are_free_and_unknown_billing_blocks_new_requests(self):
        error = json.dumps({'error': {'message': 'rate limited'}}).encode()
        provider = FakeProvider([(429, error, 'application/json'), (500, b'oops', 'text/plain')])
        with tempfile.TemporaryDirectory() as temp:
            accounting = Accounting(Path(temp), provider.url, 'k', {'gpt-6.1-sol': PRICES})
            accounting.set_context('direct', Path(temp))
            with accounting.serve() as proxy:
                self.assertEqual(post(proxy)[0], 429)
                self.assertEqual(accounting.spent(), 0)
                self.assertEqual(post(proxy)[0], 502)
                self.assertIsNone(accounting.spent())
                status, body = post(proxy)
            self.assertEqual(status, 402)
            self.assertIn(b'unknown_billing', body)
            self.assertEqual(len(provider.received), 2)

    def test_unpriced_model_is_refused_before_sending(self):
        provider = FakeProvider([])
        with tempfile.TemporaryDirectory() as temp:
            accounting = Accounting(Path(temp), provider.url, 'k', {'gpt-6.1-sol': PRICES})
            accounting.set_context('opencode', Path(temp))
            with accounting.serve() as proxy:
                status, body = post(proxy, model='openai/gpt-other')
            self.assertEqual(status, 402)
            self.assertIn(b'unpriced_model', body)
            self.assertEqual(provider.received, [])


if __name__ == '__main__':
    unittest.main()
