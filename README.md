# Smart Voice Dictation

> 🚧 **Work in progress** — nothing is installable yet. Follow the repository to know when the first release lands.

**Speak anywhere, get clean text.** A free, self-hosted alternative to Wispr Flow.

Free, open-source alternative to Wispr Flow. Hold a hotkey, speak, and clean punctuated text appears wherever your cursor is. Self-hosted Whisper transcription, LLM cleanup with free-tier fallback, voice-activated writing modes (email, AI prompt, notes) and a self-learning vocabulary. Windows first, MIT.

## Planned features (v1)

- Push-to-talk dictation in any Windows application (default hotkey: `Ctrl + Win`).
- Self-hosted Whisper transcription server (on your own server or on the same PC).
- LLM cleanup (punctuation, hesitations, self-corrections) through a configurable chain of providers, free tiers first.
- Voice-activated writing modes: default, message, email, AI prompt, notes.
- A personal vocabulary that learns from your corrections.
- Mixed-language dictation (e.g. French with English technical terms) is preserved as spoken.
- Privacy first: audio only goes to your own server and is never stored.

## Components

| Folder | What it is |
| --- | --- |
| `server/` | Transcription server (FastAPI + faster-whisper), OpenAI-compatible API, runs in Docker |
| `core/` | OS-independent logic: profiles, vocabulary, LLM chain, budget, history |
| `client-windows/` | Windows client: hotkey, audio capture, paste, overlay, tray icon |

Deploying the server: see [docs/server.md](docs/server.md).

## License

[MIT](LICENSE)
