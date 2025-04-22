
# Matrix AI Chat Bot

A fully asynchronous Python bot for [Matrix](https://matrix.org/) that connects to various vLLM (OpenAI-compatible) endpoints, logs all messages in a database for **persistent multi-turn conversation**, and provides commands for querying AI models, managing endpoints, and more.

## Features

- **Matrix-NIO Integration**: Uses `nio` (matrix-nio) to log in and sync with a Matrix server.
- **Persistent Conversation Memory**: Stores every message (user & bot) in an SQLite DB, so the bot “remembers” previous chat context across restarts.
- **Asynchronous SQLAlchemy**: Uses `sqlalchemy.ext.asyncio` and `aiosqlite` for non-blocking DB operations.
- **vLLM Endpoints**: Dynamically loads AI models from multiple endpoints (e.g., Hugging Face / custom OpenAI-like services).  
- **Model Discovery**: Each model is assigned a numeric ID, so users can query them easily.
- **Commands**:
  - `!ask <model_or_alias> <prompt> [max_tokens=?] [temp=?] [top_p=?]` – Query a model with optional parameters.
  - `!models` – List discovered models.
  - `!stats` – Show usage stats, top users, etc.
  - `!reload_models` – Owner-only reload of the model map.
  - `!add_endpoint` / `!remove_endpoint` – Trusted users can add/remove vLLM endpoints.
  - `!alias` / `!aliases` / `!rm_alias` – Create or remove aliases for model IDs.
  - `!help` – List commands.
- **Role-Based Permissions**: 
  - **Owner** commands (e.g. `!reload_models`). 
  - **Trusted users** can manage endpoints and aliases.
- **Rate Limiting**: Basic per-user limit to prevent spam.
- **Error Handling & Retries**: Automatically retries model queries if endpoints fail briefly.

## Requirements

- **Python 3.9+** (3.10 recommended)
- The following Python libraries (exact versions may vary):
  
  ```text
  matrix-nio==0.19.1
  openai==0.27.2
  sqlalchemy==1.4.46
  aiosqlite==0.18.0
  ```
  
  You can install them via:
  ```bash
  pip install -r requirements.txt
  ```
  
  *(Adjust or update versions as needed.)*

## Installation & Setup

1. **Clone or Download** this repository:

   ```bash
   git clone https://github.com/yourname/matrix-ai-chat-bot.git
   cd matrix-ai-chat-bot
   ```

2. **Install Dependencies**:

   ```bash
   pip install -r requirements.txt
   ```
   
   or install manually:
   
   ```bash
   pip install matrix-nio==0.19.1 openai==0.27.2 sqlalchemy==1.4.46 aiosqlite==0.18.0
   ```
   
3. **Configuration**:

   You can specify important credentials via environment variables (recommended) or edit them directly in the code if desired (less secure). Environment variables:

   - `MATRIX_HOMESERVER` – Base URL of your Matrix server (default: `https://matrix.org`)
   - `MATRIX_USER_ID` – Full Matrix user ID for the bot (e.g. `@mybot:matrix.org`)
   - `MATRIX_PASSWORD` – Password for the bot user
   - `MATRIX_OWNER_ID` – The single user ID allowed to run certain privileged commands (e.g. `!reload_models`)
   
   Example:
   ```bash
   export MATRIX_HOMESERVER="https://matrix.org"
   export MATRIX_USER_ID="@mybot:matrix.org"
   export MATRIX_PASSWORD="superSecretPassword"
   export MATRIX_OWNER_ID="@myowner:matrix.org"
   ```
   
4. **Run the Bot**:

   ```bash
   python matrix_bot.py
   ```
   
   Replace `matrix_bot.py` with the name of your main script (e.g., `main.py`) if necessary. The bot will:
   - Initialize the SQLite database (`data/matrix-bot.sqlite` by default).
   - Log in to your Matrix server.
   - Enter a sync loop, listening for messages indefinitely.

5. **Invite the Bot** to a Room:
   - In Matrix, invite the bot user (e.g., `@mybot:matrix.org`) to a room or direct message it.
   - The bot responds to commands in rooms or direct chats.

## Usage

Once the bot is running and has joined your room:

- **General Conversation**: Users can chat normally. All messages get stored in the DB.
- **Commands**: Start each command with the defined prefix (`!`) by default.  
  - `!ask <model_or_alias> <prompt>`: Query a model.  
  - `!stats`: Show usage stats.  
  - `!models`: List discovered models by numeric ID.  
  - `!reload_models`: Reload the vLLM model map (owner only).  
  - etc.

For example:

```
User: !models
Bot:  
  1. gpt-3.5-turbo
  2. gpt-4
User: !ask 1 Hello, how are you?
Bot: **gpt-3.5-turbo** says:
     Hello! I am an AI model...
```

### Commands Overview

| Command                         | Description                                                                         |
|--------------------------------|-------------------------------------------------------------------------------------|
| `!help`                        | Displays available commands.                                                        |
| `!models`                      | Lists the currently discovered models with numeric IDs.                              |
| `!ask <model_or_alias> <prompt> [max_tokens=?] [temp=?] [top_p=?]` | Queries the specified model with optional parameters.                            |
| `!stats`                       | Shows bot usage stats, total messages, and top 5 users from in-memory counts.       |
| `!reload_models`               | Reloads the model map (privileged to bot owner only).                                |
| `!add_endpoint <name> <url> <api_key>` (trusted)  | Dynamically adds a new vLLM endpoint, triggers an immediate reload.               |
| `!remove_endpoint <name>` (trusted)                | Removes a named endpoint and reloads models.                                       |
| `!alias <alias> <model_id>` (trusted)              | Sets an alias for a model, so `!ask alias ...` works.                              |
| `!aliases`                    | Lists the currently defined aliases.                                                |
| `!rm_alias <alias>` (trusted)  | Removes a previously set alias.                                                     |

**Rate Limiting**: If a user exceeds the permitted requests within 60 seconds, the bot replies with a rate-limit warning.

## Folder Structure

Typical layout:

```
matrix-ai-chat-bot/
├── data/
│   └── matrix-bot.sqlite        # SQLite database (generated automatically)
├── matrix_bot.py                # Main bot script
├── requirements.txt             # Python deps
└── README.md                    # This file
```

## Customizing / Extending

- **Database**:  
  - The code uses SQLite for simplicity. Switch to PostgreSQL or MySQL by adjusting the `DATABASE_URL` in the script and installing the relevant async driver (e.g. `asyncpg`).
- **Conversation Depth**:  
  - Adjust `MAX_HISTORY_LENGTH` in the code if you want more (or fewer) messages in the context.
- **Permissions**:  
  - Add or remove user IDs from the `TRUSTED_USERS` set in the code to allow them to run advanced commands.
- **Commands**:  
  - Each command is mapped to a handler function. Add your own custom command by following the existing pattern and registering it in `self.commands`.
- **AI Parameters**:  
  - `!ask` supports `max_tokens=`, `temp=`, and `top_p=`. You can parse more parameters (e.g., presence_penalty) if your vLLM service supports them.

## Troubleshooting

- **Bot Not Responding**:
  - Check that your Matrix user ID and password are correct.
  - Make sure the bot user is joined in the room.  
  - Inspect logs for errors (`python matrix_bot.py` should print out info).
- **Model Not Found**:
  - Ensure the endpoint is reachable and that `client.models.list()` actually returns model IDs.  
  - Use `!models` to confirm the model is registered.
- **Database Issues**:
  - Ensure the `data/` directory is writable or create it manually.  
  - For concurrency or large-scale usage, consider a more robust DB.

## License

- Public Domain