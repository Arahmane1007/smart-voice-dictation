# Changelog

All notable changes to this project are documented here. The format follows [Keep a Changelog](https://keepachangelog.com/) and the project uses [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- uv workspace with the `core`, `server` and `client-windows` packages, tooling and CI.
- Transcription server: OpenAI-compatible `/v1/audio/transcriptions` endpoint, API keys, rate limits, upload and duration caps, single-worker queue, JSON logs without transcribed text.
- Docker image and release workflow publishing it to GitHub Container Registry.
