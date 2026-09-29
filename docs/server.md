# Transcription server

The server receives audio, transcribes it with [faster-whisper](https://github.com/SYSTRAN/faster-whisper) and returns the text. It exposes an OpenAI-compatible endpoint, `POST /v1/audio/transcriptions`, and a `GET /health` check. It never stores audio and never logs transcribed text.

## 1. Generate an API key

```bash
docker run --rm ghcr.io/<owner>/smart-voice-dictation-server svd-server generate-key
```

Before the first image is published, generate it from a clone of the repository instead: `uv run svd-server generate-key`.

Keep it secret: anyone with this key can use your server.

## 2a. Run it on the same PC

```bash
cd server
API_KEYS=<your-key> docker compose up -d --build
curl http://127.0.0.1:8000/health    # "ok" once the model is loaded
```

Plain HTTP is only acceptable on `localhost`; the client refuses it for any other address.

## 2b. Run it on a server with a domain (example: Dokploy)

1. Create a **Compose** application pointing to this repository, compose path `server/docker-compose.yml` (or use the published image directly).
2. In *Environment*, set `API_KEYS=<your-key>` and `GITHUB_OWNER=<owner>`.
3. In *Domains*, attach your domain to port `8000` with HTTPS enabled (Let's Encrypt).
4. Set `TRUSTED_PROXIES` to the network of the reverse proxy, so rate limits see the real client IP. With Dokploy, find it with `docker network inspect dokploy-network` (field `Subnet`, e.g. `10.0.1.0/24`).
5. On Dokploy the domain is routed by Traefik to container port `8000`, so the published `127.0.0.1:8000:8000` mapping is not needed there; remove it from the Dokploy app if it collides with something else on port 8000.
6. Deploy, then check: `curl https://your-domain/health` → `ok`.

## Configuration

| Variable | Default | Meaning |
| --- | --- | --- |
| `API_KEYS` | *(required)* | Accepted keys, comma-separated, at least 32 characters each |
| `WHISPER_MODEL` | `small` | Model size (`tiny`, `base`, `small`, `medium`, `large-v3`…) |
| `WHISPER_COMPUTE_TYPE` | `int8` | Precision (`int8` is best on CPU) |
| `WHISPER_BEAM_SIZE` | `5` | Higher is more accurate, lower is faster |
| `CPU_THREADS` | `0` | `0` uses every core |
| `MAX_UPLOAD_MB` | `10` | Maximum upload size |
| `MAX_AUDIO_SECONDS` | `125` | Maximum audio duration |
| `RATE_LIMIT_PER_MINUTE` | `30` | Requests per minute and per key |
| `AUTH_FAILURES_PER_MINUTE` | `10` | Failed authentications tolerated per minute and per IP |
| `QUEUE_SIZE` | `2` | Requests allowed to wait while one is transcribed |
| `TRUSTED_PROXIES` | *(empty)* | IPs or networks allowed to set `X-Forwarded-For` |
| `MEM_LIMIT` | `4g` | Container memory cap (Docker Compose `mem_limit`). The `small` model needs about 1–2 GB; allow more for `medium`/`large-v3` |

## Rotating a key without downtime

1. Add the new key: `API_KEYS=<old>,<new>`, redeploy.
2. Put the new key in your client.
3. Remove the old key: `API_KEYS=<new>`, redeploy.

## Testing with curl

```bash
curl -s https://your-domain/v1/audio/transcriptions \
  -H "Authorization: Bearer <your-key>" \
  -F file=@clip.m4a -F language=fr -F response_format=verbose_json
```
