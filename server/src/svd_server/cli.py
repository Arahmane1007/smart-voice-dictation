"""Command line: `svd-server serve` and `svd-server generate-key`."""

import argparse
import os
import secrets
import sys

import uvicorn

from svd_server.app import create_app
from svd_server.engine import FasterWhisperEngine
from svd_server.logging_setup import configure_logging
from svd_server.settings import Settings, SettingsError


def generate_key() -> str:
    return secrets.token_urlsafe(32)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="svd-server")
    commands = parser.add_subparsers(dest="command", required=True)
    serve = commands.add_parser("serve", help="run the transcription server")
    serve.add_argument("--host", default="0.0.0.0")
    serve.add_argument("--port", type=int, default=8000)
    commands.add_parser("generate-key", help="print a new random API key")
    args = parser.parse_args(argv)

    if args.command == "generate-key":
        print(generate_key())
        return 0

    try:
        settings = Settings.from_env(os.environ)
    except SettingsError as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return 2

    configure_logging()
    engine = FasterWhisperEngine(settings)
    # proxy_headers=False: X-Forwarded-For is handled by our own trusted-proxy logic.
    uvicorn.run(
        create_app(settings, engine),
        host=args.host,
        port=args.port,
        proxy_headers=False,
        access_log=False,
    )
    return 0
