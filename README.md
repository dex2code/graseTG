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
- A question asked in **reply to another participant's message** (with an
  @mention) carries that message as a clearly marked quote — the model sees
  what the question is about ("@bot what did he mean?" while replying to a
  recipe). A reply to another participant WITHOUT an @mention is still
  ignored — plain chat must stay silent.
- Edited messages are not re-answered — only new messages are processed.
- Only **text questions** are answered by the LLM. Addressed photos, videos,
  GIFs and documents are answered with a "can't analyze attachments yet" stub.
  Voice/audio and stickers are addressed only via a reply to the bot — they
  get a "please type your question" stub answer. The content pipeline is
  extensible — new types are added as extractors in `src/grase_tg/content.py`.
- Each question is sent to an OpenAI-compatible API without chat history;
  long answers are split into chunks.
- The current date is injected into every request ("tomorrow" must resolve
  to a real date). If a `tools:` section is configured, the model may call
  tools exposed by one MCP server — see "External tools (MCP)" below.
- No memory: every question is processed independently. The two exceptions:
  a reply to a bot answer carries that answer into the request as dialogue
  context, so follow-up questions ("and what about chicken?") keep their
  referent; a reply to another participant's message (with an @mention)
  carries that message as a marked quote inside the question. Replies to the
  ack, stubs and error notices carry no context; replying to one chunk of a
  long answer carries only that chunk.
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
   - `MCP_TOKEN` — (optional) bearer token for the MCP tool server: name it
     in `config.yaml` via `tools.mcp.auth_env`;
   - `TELEGRAM_PROXY_URL` — (optional) proxy for Telegram traffic:
     `http://proxy.local:8080` or `socks5://proxy.local:1080`, with auth
     `scheme://user:password@proxy.local:port`. Supported schemes: `http`,
     `socks4`, `socks5`; an invalid URL is rejected at startup.
     The proxy applies to Telegram traffic only; LLM requests go direct.
2. **Settings** — copy `config.example.yaml` to `config.yaml` (gitignored) and adjust:
   - `allowed_chat_ids` — IDs of the allowed groups;
   - `llm.model` — the model name on your backend;
   - `llm.temperature`, optional sampling overrides (`llm.top_p`,
     `llm.presence_penalty`), `llm.max_tokens`, `llm.timeout_seconds` —
     generation parameters and the per-request HTTP timeout; an unset
     override is not sent at all, so the backend applies its own default;
   - `llm.enable_thinking` — model "thinking" toggle (`chat_template_kwargs.enable_thinking`
     for vLLM/SGLang-style backends, Qwen3, etc.): `true`/`false` send the field,
     `null` omits it entirely (safest for plain OpenAI APIs);
   - `llm.tool_rounds` — max tool-call rounds per question (default 2, must be ≥ 1);
   - `tools.mcp` — the single MCP tool server (streamable HTTP): `url`,
     optional `auth_env` (a .env variable NAME holding the token),
     `timeout_seconds`, optional `allow` whitelist. Absent → tools disabled;
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

## External tools (MCP)

With a `tools:` section in `config.yaml` the bot bridges **one MCP server**
(streamable HTTP) into OpenAI tool calling: the model decides when a question
needs external data (weather, exchange rates, ...), the bot executes the call
and feeds the result back — up to `llm.tool_rounds` rounds per question.
Knowledge questions (recipes, ...) never trigger tools.

Requirements on the backend: llama-server with tool calling (run it with
`--jinja`) and a tool-capable model (Qwen3, Llama 3.x, ...).

- Exactly one server, by design: `tools.mcp` is a single block, not a list —
  describing a second server is impossible (see `docs/mcp-tools-spec.md`).
- Tool arguments are validated by the MCP server; a failing tool never
  crashes the answer — the error text goes back to the model, which words
  the apology.
- If the server is unreachable at startup, the bot logs a warning and runs
  without tools. A broken config (bad `url`, missing `auth_env` variable)
  is a clean startup failure.
- Tool calls take extra time: worst case ~`tool_rounds` × (LLM + tool
  timeout). The "Got it, thinking…" ack covers the wait.
- Privacy: tool names and argument counts go to the logs; argument values
  and question texts do not.

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
