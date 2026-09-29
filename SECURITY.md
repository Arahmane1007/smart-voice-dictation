# Security policy

This application installs a global keyboard hook, reads and writes the clipboard and sends audio to a server. Security reports are taken seriously.

## Reporting a vulnerability

Please **do not open a public issue**. Use GitHub's private vulnerability reporting instead: *Security* tab → *Report a vulnerability*.

Include what you found, how to reproduce it and the affected version or commit. This is a personal open-source project maintained on a best-effort basis; you will get an answer as soon as possible.

## Scope

- The transcription server (`server/`): authentication, rate limits, upload limits, logs.
- The Windows client (`client-windows/`): keyboard hook, clipboard handling, stored secrets.
