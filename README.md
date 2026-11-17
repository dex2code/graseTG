# graseTG

[![CI](https://github.com/dex2code/graseTG/actions/workflows/ci.yml/badge.svg)](https://github.com/dex2code/graseTG/actions/workflows/ci.yml)

A family Telegram bot: answers questions in whitelisted group chats via an LLM.

## How it works

- The bot answers **only in whitelisted groups** (`config.yaml` → `allowed_chat_ids`).
- The trigger is one of:
  - an `@mention` of the bot with the question in the same message:
    `@grase_bot how do I cook borscht?`;
  - a **reply** to any of the bot's messages: reply to an answer and keep
    asking in plain text.
- Private messages and unknown groups are silently ignored.
- The bot reacts **only to messages addressed to it**: an `@mention` in the
  message text or in a media caption (photo, video, GIF, document), or a reply
  to one of the bot's messages. Unaddressed messages are silently ignored.
  A mention and a reply combined: the mention shapes the question text, the
  reply still contributes its context.
- Edited messages are not re-answered — only new messages are processed.
- Only **text questions** are answered by the LLM. Addressed photos, videos,
  GIFs and documents are answered with a "can't analyze attachments yet" stub.
  Voice/audio and stickers are addressed only via a reply to the bot — they
  get a "please type your question" stub answer. The content pipeline is
  extensible — new types are added as extractors in `src/grase_tg/content.py`.
- Each question is sent to an OpenAI-compatible API without chat history;
  long answers are split into chunks.
- No memory: every question is processed independently. The one exception:
  a reply to a bot answer carries that answer into the request as dialogue
  context, so follow-up questions ("and what about chicken?") keep their
  referent. Replies to the ack, stubs and error notices carry no context;
  replying to one chunk of a long answer carries only that chunk.
- Privacy: question texts are never written to logs — only the chat id and
  the question length.
- Updates that arrived while the bot was offline are discarded on startup —
  no delayed replies to stale mentions after a restart.
- The bot acknowledges immediately ("Got it, thinking…"), then edits that message
  with the model's answer. If the model fails, the same message shows the error;
  if editing fails, the answer (or the error) is re-sent as a fresh reply.
  Replies to the pending ack are skipped — wait for the answer, then reply to it.

## Setup

1. **Secrets** — copy `.env.example` to `.env` and fill it in:
   - `BOT_TOKEN` — the token from [@BotFather](https://t.me/BotFather);
   - `LLM_BASE_URL` — an OpenAI-compatible API base URL (usually ends with `/v1`);
   - `LLM_API_KEY` — the API access key;
   - `TELEGRAM_PROXY_URL` — (optional) proxy for Telegram traffic:
     `http://proxy.local:8080` or `socks5://proxy.local:1080`, with auth
     `scheme://user:password@proxy.local:port`. Supported schemes: `http`,
     `socks4`, `socks5`; an invalid URL is rejected at startup.
     The proxy applies to Telegram traffic only; LLM requests go direct.
2. **Settings** — copy `config.example.yaml` to `config.yaml` (gitignored) and adjust:
   - `allowed_chat_ids` — IDs of the allowed groups;
   - `llm.model` — the model name on your backend;
   - `llm.temperature`, `llm.max_tokens`, `llm.timeout_seconds` — generation
     parameters and the per-request HTTP timeout;
   - `llm.enable_thinking` — model "thinking" toggle (`chat_template_kwargs.enable_thinking`
     for vLLM/SGLang-style backends, Qwen3, etc.): `true`/`false` send the field,
     `null` omits it entirely (safest for plain OpenAI APIs);
   - `system_prompt` — answer style and rules (replace the placeholder with your own);
   - `messages` — the bot's user-facing texts (ack, errors, stubs). Defaults
     are English — translate them to your language here.
   - Unknown keys are rejected at startup — a typo fails fast instead of
     silently disabling a setting.

The bot looks for `.env` and `config.yaml` in its **working directory**
(set `GRASE_TG_CONFIG_DIR` to point elsewhere), so run it from the project
root — `uv run grase-tg` does that by default.

## How to find a group ID

1. Add the bot to a group and post any `@mention` of it there.
2. A line like this appears in the logs:
   `Chat 'Our Family' (id=-1001234567890) is not whitelisted — ignoring.`
3. Put that id into `allowed_chat_ids` in `config.yaml` and restart the bot.

> Privacy mode must be enabled for the bot (the default from @BotFather) —
> that way Telegram only delivers `@mentions`, commands and replies to the bot.

## Running

```bash
uv sync          # create .venv and install dependencies
uv run grase-tg  # start the bot
```

## Deploying as a service (systemd)

The repo ships a ready-made unit: `graseTG.service`. It runs the bot's venv
entry point directly (`.venv/bin/grase-tg` — no uv needed at runtime), reads
secrets and settings from the working directory, restarts on crashes (but
stays down after 5 crashes per minute), and applies basic hardening. See the
header of the unit file for the step-by-step deployment checklist.

## Testing

```bash
uv run pytest  # unit tests: content extraction, handlers, LLM payload, config, startup
```

All tests are offline: the LLM is a local stub server, Telegram objects are fakes.
