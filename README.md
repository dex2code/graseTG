# graseTG

A family Telegram bot: answers questions in whitelisted group chats via an LLM.

## How it works

- The bot answers **only in whitelisted groups** (`config.yaml` → `allowed_chat_ids`).
- The trigger is an `@mention` of the bot with the question in the same message:
  `@grase_bot how do I cook borscht?`
- Private messages and unknown groups are silently ignored.
- Each question is sent without chat history to an OpenAI-compatible API;
  the answer arrives as a reply.
- No memory: every question is processed independently.
- The bot acknowledges immediately ("Got it, thinking…"), then edits that message
  with the model's answer. If the model fails, the same message shows the error.

## Setup

1. **Secrets** — copy `.env.example` to `.env` and fill it in:
   - `BOT_TOKEN` — the token from [@BotFather](https://t.me/BotFather);
   - `LLM_BASE_URL` — an OpenAI-compatible API base URL (usually ends with `/v1`);
   - `LLM_API_KEY` — the API access key;
   - `TELEGRAM_PROXY_URL` — (optional) HTTP proxy for Telegram:
     `http://proxy.local:8080` or with auth `http://user:password@proxy.local:8080`.
     The proxy applies to Telegram traffic only; LLM requests go direct.
2. **Settings** — `config.yaml`:
   - `allowed_chat_ids` — IDs of the allowed groups;
   - `llm.model` — the model name on your backend;
   - `llm.enable_thinking` — model "thinking" toggle (`chat_template_kwargs.enable_thinking`
     for vLLM/SGLang-style backends, Qwen3, etc.);
   - `system_prompt` — answer style and rules (replace the placeholder with your own).

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
