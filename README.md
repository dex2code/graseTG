# graseTG

[![CI](https://github.com/dex2code/graseTG/actions/workflows/ci.yml/badge.svg)](https://github.com/dex2code/graseTG/actions/workflows/ci.yml)

A family Telegram bot: answers questions in whitelisted group chats via an LLM.

## How it works

- The bot answers **only in whitelisted groups** (`config.yaml` → `allowed_chat_ids`).
- The trigger is an `@mention` of the bot with the question in the same message:
  `@grase_bot how do I cook borscht?`
- Private messages and unknown groups are silently ignored.
- The bot reacts **only to messages addressed to it**: an `@mention` in the
  message text, or in a photo caption. Unaddressed messages are silently ignored.
- Only **text questions** are answered by the LLM. An addressed photo is answered
  with a "can't analyze images yet" stub. Handlers for audio and stickers are
  declared but dormant — these types cannot carry an @mention yet. The content
  pipeline is extensible — new types are added as extractors in `src/grase_tg/content.py`.
- Each question is sent without chat history to an OpenAI-compatible API;
  long answers are split into chunks.
- No memory: every question is processed independently.
- Updates that arrived while the bot was offline are discarded on startup —
  no delayed replies to stale mentions after a restart.
- The bot acknowledges immediately ("Got it, thinking…"), then edits that message
  with the model's answer. If the model fails, the same message shows the error.

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
   - `llm.enable_thinking` — model "thinking" toggle (`chat_template_kwargs.enable_thinking`
     for vLLM/SGLang-style backends, Qwen3, etc.): `true`/`false` send the field,
     `null` omits it entirely (safest for plain OpenAI APIs);
   - `system_prompt` — answer style and rules (replace the placeholder with your own).
   - Unknown keys are rejected at startup — a typo fails fast instead of
     silently disabling a setting.

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

The repo ships a ready-made unit: `graseTG.service`. It runs the bot via
`uv run --frozen grase-tg`, restarts on crashes (but stays down after 5 crashes
per minute), and applies basic hardening. See the header of the unit file for
the step-by-step deployment checklist.

## Testing

```bash
uv run pytest  # unit tests: content extraction, handlers, LLM payload, config
```

All tests are offline: the LLM is a local stub server, Telegram objects are fakes.
