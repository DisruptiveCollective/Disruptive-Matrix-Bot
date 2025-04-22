#!/usr/bin/env python3
import os
import logging
import asyncio
import pathlib
import signal
import time
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any

# matrix-nio imports
from nio import AsyncClient, MatrixRoom, RoomMessageText, LoginResponse

# OpenAI-compatible client (assumed installed)
from openai import OpenAI

# SQLAlchemy async imports
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy import select, Column, Integer, String, Boolean, DateTime
from sqlalchemy.orm import sessionmaker, declarative_base

##############################################################################
# CONFIGURATION & LOGGING
##############################################################################

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=LOG_LEVEL,
    format="[%(asctime)s] %(levelname)s:%(name)s: %(message)s"
)
logger = logging.getLogger("MatrixBot")

DATA_DIR = pathlib.Path("data")
DATA_DIR.mkdir(exist_ok=True)

# Example: read from env vars or use fallback values
MATRIX_HOMESERVER = os.getenv("MATRIX_HOMESERVER", "https://matrix.org")
MATRIX_USER_ID = os.getenv("MATRIX_USER_ID", "@disruptiveai:matrix.org")
MATRIX_PASSWORD = os.getenv("MATRIX_PASSWORD", "Z5i!thXxZ7EwWX3")
MATRIX_OWNER_ID = os.getenv("MATRIX_OWNER_ID", "@thedisruptivecollective:matrix.org")

TRUSTED_USERS = {
    MATRIX_OWNER_ID,
    "@some-other-moderator:matrix.org"
}

# Async DB URL (using aiosqlite for example)
DATABASE_URL = "sqlite+aiosqlite:///data/matrix-bot.sqlite"

# Hardcoded vLLM endpoints
HARDCODED_VLLM_ENDPOINTS = [
    {"name": "endpoint1", "base_url": "https://hermes.ai.unturf.com/v1", "api_key": "not-needed"},
    {"name": "endpoint2", "base_url": "https://naptha2.ai.unturf.com/v1", "api_key": "not-needed"},
    {"name": "endpoint3", "base_url": "https://naptha3.ai.unturf.com/v1", "api_key": "not-needed"},
]

##############################################################################
# SQLALCHEMY SETUP (ASYNC)
##############################################################################

Base = declarative_base()

class Message(Base):
    """
    SQLAlchemy model to store Matrix messages (both user and bot).
    We'll rely on these records for conversation context.
    """
    __tablename__ = "messages"

    id = Column(Integer, primary_key=True, index=True)
    room_id = Column(String, nullable=True)
    user_id = Column(String, nullable=True)
    content = Column(String, nullable=False)
    timestamp = Column(DateTime, nullable=False)
    is_bot = Column(Boolean, default=False)
    model_name = Column(String, nullable=True)

async_engine = create_async_engine(DATABASE_URL, echo=False, future=True)
AsyncSessionLocal = sessionmaker(
    bind=async_engine,
    expire_on_commit=False,
    class_=AsyncSession
)

async def init_db():
    """
    Create the database schema if it doesn't exist yet.
    """
    async with async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("Database initialized (async).")

async def save_message(
    room_id: str,
    user_id: str,
    content: str,
    is_bot: bool,
    model_name: Optional[str] = None
):
    """
    Asynchronously save a message record to the DB. 
    We'll query from this table to build context for the bot.
    """
    now = datetime.utcnow()
    try:
        async with AsyncSessionLocal() as session:
            async with session.begin():
                msg = Message(
                    room_id=room_id,
                    user_id=user_id,
                    content=content,
                    timestamp=now,
                    is_bot=is_bot,
                    model_name=model_name,
                )
                session.add(msg)
        logger.debug(f"Saved message: {room_id} => {content[:50]}...")
    except Exception as e:
        logger.error(f"Failed to save message: {e}", exc_info=True)

##############################################################################
# vLLM ENDPOINT DISCOVERY & MODEL MAP INITIALIZATION
##############################################################################

MODEL_CLIENT_MAP: Dict[str, OpenAI] = {}       # model_id -> OpenAI client instance
MODEL_ID_TO_NUMBER: Dict[str, int] = {}          # full model_id -> numeric ID
NUMBER_TO_MODEL_ID: Dict[int, str] = {}          # numeric ID -> full model_id

DYNAMIC_VLLM_ENDPOINTS: List[Dict[str, str]] = []  # endpoints added at runtime
model_map_lock = asyncio.Lock()  # concurrency lock for reloading models

def load_vllm_endpoints() -> List[Dict[str, str]]:
    """Return the union of hardcoded and dynamic endpoints."""
    return HARDCODED_VLLM_ENDPOINTS + DYNAMIC_VLLM_ENDPOINTS

async def initialize_model_map():
    """
    Clears existing model maps and re-discovers vLLM endpoints.
    Lock-protected so multiple commands don't collide.
    """
    async with model_map_lock:
        MODEL_CLIENT_MAP.clear()
        MODEL_ID_TO_NUMBER.clear()
        NUMBER_TO_MODEL_ID.clear()

        discovered = load_vllm_endpoints()
        if not discovered:
            logger.warning("No vLLM endpoints provided.")
            return

        for ep in discovered:
            name = ep["name"]
            base_url = ep["base_url"]
            api_key = ep["api_key"]
            logger.info(f"Loading models from '{name}' => {base_url} (api_key={api_key})")

            client = OpenAI(base_url=base_url, api_key=api_key)
            try:
                # Wrap the blocking call in asyncio.to_thread
                resp = await asyncio.to_thread(lambda: client.models.list())
                model_list = resp.data
                logger.info(f"Found {len(model_list)} models on '{name}'")
            except Exception as e:
                logger.warning(f"Could not list models for '{name}': {e}", exc_info=True)
                continue

            for m in model_list:
                model_id = m.id
                if model_id and (model_id not in MODEL_CLIENT_MAP):
                    MODEL_CLIENT_MAP[model_id] = client
                    logger.info(f"Registered model '{model_id}'")

        sorted_ids = sorted(MODEL_CLIENT_MAP.keys())
        num = 1
        for mid in sorted_ids:
            MODEL_ID_TO_NUMBER[mid] = num
            NUMBER_TO_MODEL_ID[num] = mid
            num += 1

        logger.info("Finished building vLLM model map with numeric IDs:")
        for num_id, model in NUMBER_TO_MODEL_ID.items():
            logger.info(f"  {num_id} => {model}")

def get_client_for_model_id(model_id: str) -> Optional[OpenAI]:
    """Return the client for a given model ID, if available."""
    return MODEL_CLIENT_MAP.get(model_id)

##############################################################################
# HELPER CLASSES: PERMISSIONS, RATE LIMITING
##############################################################################

def is_owner(user_id: str) -> bool:
    return user_id == MATRIX_OWNER_ID

def is_trusted(user_id: str) -> bool:
    return user_id in TRUSTED_USERS

class RateLimiter:
    """
    Simple rate limiting: max N requests per user in last 60 seconds.
    """
    def __init__(self, max_requests_per_minute: int = 50):
        self.max_requests_per_minute = max_requests_per_minute
        self.user_request_timestamps: Dict[str, List[float]] = {}

    def check(self, user_id: str) -> bool:
        now = time.time()
        if user_id not in self.user_request_timestamps:
            self.user_request_timestamps[user_id] = []

        # prune old timestamps
        self.user_request_timestamps[user_id] = [
            ts for ts in self.user_request_timestamps[user_id] if (now - ts) < 60
        ]
        if len(self.user_request_timestamps[user_id]) >= self.max_requests_per_minute:
            return False

        self.user_request_timestamps[user_id].append(now)
        return True

##############################################################################
# MODEL QUERY WITH RETRIES
##############################################################################

async def query_model_with_retries(
    client: OpenAI,
    model_id: str,
    messages: List[Dict[str, str]],
    max_tokens: int = 256,
    temperature: float = 0.5,
    top_p: float = 1.0,
    attempts: int = 3
) -> Any:
    """
    Try up to `attempts` times to query the model, with async sleep backoffs if errors occur.
    Uses asyncio.to_thread to avoid blocking the event loop.
    """
    for i in range(attempts):
        try:
            # Wrap the blocking API call in asyncio.to_thread
            resp = await asyncio.to_thread(
                lambda: client.chat.completions.create(
                    model=model_id,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                )
            )
            return resp
        except Exception as e:
            logger.error(f"Error querying model '{model_id}' (attempt {i+1}): {e}")
            if i < attempts - 1:
                await asyncio.sleep(1)
            else:
                raise e

##############################################################################
# THE BOT
##############################################################################

class MatrixBot:
    """
    Main bot class:
     - Uses AsyncClient for Matrix.
     - Processes commands.
     - Implements rate limiting.
     - Persists conversation memory in a DB.
     - Manages model aliasing.
    """

    MAX_HISTORY_LENGTH = 5

    def __init__(self, homeserver: str, user_id: str, password: str):
        self.client = AsyncClient(homeserver, user_id)
        self.password = password
        self.command_prefix = "!"
        self.start_time = datetime.utcnow()  # For uptime tracking

        # Aliases for models: alias -> real_model_id
        self.model_aliases: Dict[str, str] = {}

        # Rate limiter
        self.rate_limiter = RateLimiter(50)

        # Track usage per user
        self.user_message_count: Dict[str, int] = {}

        # Define command -> handler mapping
        self.commands: Dict[str, Any] = {
            "models": self.cmd_models,
            "ask": self.cmd_ask,
            "stats": self.cmd_stats,
            "reload_models": self.cmd_reload_models,
            "add_endpoint": self.cmd_add_endpoint,
            "remove_endpoint": self.cmd_remove_endpoint,
            "alias": self.cmd_alias,
            "aliases": self.cmd_aliases,
            "rm_alias": self.cmd_remove_alias,
            "help": self.cmd_help,
        }

    async def login(self):
        """Login to Matrix. Raises if login fails."""
        try:
            response = await self.client.login(self.password)
            if isinstance(response, LoginResponse):
                logger.info(f"Logged in as {self.client.user}")
            else:
                logger.error(f"Login failed: {response}")
                raise Exception("Matrix login failed")
        except Exception as e:
            logger.exception("Exception during login")
            raise e

    async def run(self):
        """Log in and then start sync_forever()."""
        await self.login()
        logger.info("Starting sync loop...")
        try:
            await self.client.sync_forever(timeout=30000, full_state=True)
        except Exception as e:
            logger.exception("Error in sync_forever")
            raise e

    async def on_message(self, room: MatrixRoom, event: RoomMessageText):
        """Handle incoming messages (commands or conversation)."""
        if event.sender == self.client.user:
            return  # ignore self

        text = event.body.strip()
        logger.debug(f"Message in {room.display_name} from {event.sender}: {text[:100]}")

        # Save incoming message to DB for persistent memory
        await save_message(room.room_id, event.sender, text, is_bot=False)

        # Rate limit check
        if not self.rate_limiter.check(event.sender):
            await self.send_message(room.room_id, "Rate limit exceeded. Please wait a bit.")
            return

        # Track usage
        self.user_message_count[event.sender] = self.user_message_count.get(event.sender, 0) + 1

        # Check if this is a command
        if text.startswith(self.command_prefix):
            parts = text[len(self.command_prefix):].split()
            if not parts:
                return
            command = parts[0].lower()
            args = parts[1:]
            handler = self.commands.get(command)
            if handler:
                try:
                    await handler(room, event, args)
                except Exception as e:
                    logger.exception("Error in command handler '%s'", command)
                    await self.send_message(
                        room.room_id,
                        f"Error executing command '{command}': {str(e)}"
                    )
            else:
                await self.send_message(room.room_id, f"Unknown command: {command}")

    async def send_message(self, room_id: str, message: str):
        """Send a text message to the room and store it in the DB."""
        try:
            await self.client.room_send(
                room_id,
                message_type="m.room.message",
                content={"msgtype": "m.text", "body": message}
            )
            await save_message(room_id, "MatrixBot", message, is_bot=True)
        except Exception as e:
            logger.error(f"Failed to send message to {room_id}: {e}", exc_info=True)

    ###########################################################################
    # DB-BASED CONVERSATION MEMORY
    ###########################################################################

    async def build_chat_prompt_db(self, room_id: str) -> List[Dict[str, str]]:
        """
        Reads the last 2*MAX_HISTORY_LENGTH messages from the DB for this room in ascending
        order, skipping commands. Returns a list of messages in OpenAI chat format.
        """
        system_content = "You are a helpful vLLM assistant."
        messages = [{"role": "system", "content": system_content}]

        async with AsyncSessionLocal() as session:
            result = await session.execute(
                select(Message)
                .where(Message.room_id == room_id)
                .order_by(Message.timestamp.desc())
                .limit(self.MAX_HISTORY_LENGTH * 2)
            )
            rows = result.scalars().all()

        rows.reverse()  # oldest first

        for row in rows:
            if row.content.startswith(self.command_prefix):
                continue
            role = "assistant" if row.is_bot else "user"
            messages.append({"role": role, "content": row.content})

        return messages

    ###########################################################################
    # Parsing Utility
    ###########################################################################

    def parse_kv_params(self, tokens: List[str]) -> (List[str], Dict[str, str]):
        """
        Scan tokens for key=value pairs and return a tuple of clean tokens and the key-value dictionary.
        """
        kv_dict = {}
        clean_tokens = []
        for token in tokens:
            if '=' in token:
                key, val = token.split('=', 1)
                kv_dict[key] = val
            else:
                clean_tokens.append(token)
        return clean_tokens, kv_dict

    ###########################################################################
    # COMMAND HANDLERS
    ###########################################################################

    async def cmd_help(self, room: MatrixRoom, event: RoomMessageText, args: List[str]):
        help_text = (
            "Available Commands:\n"
            f"{self.command_prefix}help\n"
            f"{self.command_prefix}models\n"
            f"{self.command_prefix}ask <model_or_alias> <prompt> [max_tokens=?] [temp=?] [top_p=?]\n"
            f"{self.command_prefix}stats\n"
            f"{self.command_prefix}reload_models (owner)\n"
            f"{self.command_prefix}add_endpoint <name> <base_url> <api_key> (trusted)\n"
            f"{self.command_prefix}remove_endpoint <name> (trusted)\n"
            f"{self.command_prefix}alias <alias> <model_id> (trusted)\n"
            f"{self.command_prefix}aliases\n"
            f"{self.command_prefix}rm_alias <alias> (trusted)\n"
        )
        await self.send_message(room.room_id, help_text)

    async def cmd_models(self, room: MatrixRoom, event: RoomMessageText, args: List[str]):
        if not NUMBER_TO_MODEL_ID:
            await self.send_message(room.room_id, "No vLLM models discovered.")
            return
        lines = ["**Discovered Models:**"]
        for num in sorted(NUMBER_TO_MODEL_ID.keys()):
            lines.append(f"{num}. {NUMBER_TO_MODEL_ID[num]}")
        await self.send_message(room.room_id, "\n".join(lines))

    async def cmd_stats(self, room: MatrixRoom, event: RoomMessageText, args: List[str]):
        """Show usage stats, including uptime and top users."""
        model_count = len(MODEL_CLIENT_MAP)
        total_messages = 0
        try:
            async with AsyncSessionLocal() as session:
                result = await session.execute("SELECT COUNT(*) FROM messages")
                total_messages = result.scalar() or 0
        except Exception as e:
            logger.error(f"Error counting messages: {e}")

        sorted_users = sorted(self.user_message_count.items(), key=lambda x: x[1], reverse=True)
        top_users = sorted_users[:5]
        top_users_str = "\n".join([f" - {u[0]}: {u[1]} messages" for u in top_users]) if top_users else "No data"

        uptime_td: timedelta = datetime.utcnow() - self.start_time
        uptime_str = str(uptime_td).split(".")[0]  # drop microseconds

        stats_text = (
            f"**Matrix Bot Stats:**\n"
            f"- Models loaded: {model_count}\n"
            f"- Total messages logged: {total_messages}\n"
            f"- Uptime: {uptime_str}\n"
            f"**Top 5 users (in-memory counts):**\n{top_users_str}"
        )
        await self.send_message(room.room_id, stats_text)

    async def cmd_reload_models(self, room: MatrixRoom, event: RoomMessageText, args: List[str]):
        """Reload the vLLM model map. Owner only."""
        if not is_owner(event.sender):
            await self.send_message(room.room_id, "Only the bot owner can use this command.")
            return
        try:
            await initialize_model_map()
            await self.send_message(room.room_id, "vLLM model map reloaded successfully.")
        except Exception as e:
            logger.exception("Error reloading model map")
            await self.send_message(room.room_id, f"Failed to reload model map: {e}")

    async def cmd_add_endpoint(self, room: MatrixRoom, event: RoomMessageText, args: List[str]):
        """Add a new vLLM endpoint (trusted users)."""
        if not is_trusted(event.sender):
            await self.send_message(room.room_id, "Only trusted users can add endpoints.")
            return
        if len(args) < 3:
            await self.send_message(room.room_id, "Usage: !add_endpoint <name> <base_url> <api_key>")
            return

        name, base_url, api_key = args[0], args[1], args[2]
        DYNAMIC_VLLM_ENDPOINTS.append({
            "name": name,
            "base_url": base_url,
            "api_key": api_key
        })
        await initialize_model_map()
        await self.send_message(room.room_id, f"Endpoint '{name}' added and models reloaded.")

    async def cmd_remove_endpoint(self, room: MatrixRoom, event: RoomMessageText, args: List[str]):
        """Remove a previously added vLLM endpoint (trusted)."""
        if not is_trusted(event.sender):
            await self.send_message(room.room_id, "Only trusted users can remove endpoints.")
            return
        if len(args) < 1:
            await self.send_message(room.room_id, "Usage: !remove_endpoint <name>")
            return

        name = args[0]
        before_count = len(DYNAMIC_VLLM_ENDPOINTS)
        DYNAMIC_VLLM_ENDPOINTS[:] = [ep for ep in DYNAMIC_VLLM_ENDPOINTS if ep["name"] != name]
        after_count = len(DYNAMIC_VLLM_ENDPOINTS)
        if after_count == before_count:
            await self.send_message(room.room_id, f"No endpoint found with name '{name}'.")
        else:
            await initialize_model_map()
            await self.send_message(room.room_id, f"Endpoint '{name}' removed and models reloaded.")

    async def cmd_alias(self, room: MatrixRoom, event: RoomMessageText, args: List[str]):
        """Create an alias for a model ID (trusted)."""
        if not is_trusted(event.sender):
            await self.send_message(room.room_id, "Only trusted users can create model aliases.")
            return
        if len(args) < 2:
            await self.send_message(room.room_id, "Usage: !alias <alias> <model_id>")
            return

        alias_name, model_id = args[0], args[1]
        self.model_aliases[alias_name] = model_id
        await self.send_message(room.room_id, f"Alias '{alias_name}' set to '{model_id}'.")

    async def cmd_aliases(self, room: MatrixRoom, event: RoomMessageText, args: List[str]):
        """List all model aliases."""
        if not self.model_aliases:
            await self.send_message(room.room_id, "No aliases defined.")
            return
        lines = ["**Model Aliases:**"]
        for alias, real_id in self.model_aliases.items():
            lines.append(f"{alias} -> {real_id}")
        await self.send_message(room.room_id, "\n".join(lines))

    async def cmd_remove_alias(self, room: MatrixRoom, event: RoomMessageText, args: List[str]):
        """Remove a model alias (trusted)."""
        if not is_trusted(event.sender):
            await self.send_message(room.room_id, "Only trusted users can remove aliases.")
            return
        if len(args) < 1:
            await self.send_message(room.room_id, "Usage: !rm_alias <alias>")
            return

        alias_name = args[0]
        if alias_name in self.model_aliases:
            del self.model_aliases[alias_name]
            await self.send_message(room.room_id, f"Alias '{alias_name}' removed.")
        else:
            await self.send_message(room.room_id, f"No alias found with name '{alias_name}'.")

    async def cmd_ask(self, room: MatrixRoom, event: RoomMessageText, args: List[str]):
        """
        Query a vLLM model with optional persistent context from DB.
        Usage: !ask <model_or_alias> <prompt> [max_tokens=?] [temp=?] [top_p=?]
        """
        if len(args) < 2:
            await self.send_message(
                room.room_id,
                "Usage: !ask <model_or_alias> <prompt> [max_tokens=?] [temp=?] [top_p=?]"
            )
            return

        model_arg = args[0]
        rest = args[1:]
        cleaned_rest, kv_dict = self.parse_kv_params(rest)
        prompt = " ".join(cleaned_rest).strip()

        max_tokens = int(kv_dict.get("max_tokens", 1024))
        temperature = float(kv_dict.get("temp", kv_dict.get("temperature", 0.5)))
        top_p = float(kv_dict.get("top_p", 1.0))

        # Determine the real model_id from number, alias, or raw input
        if model_arg.isdigit():
            num_id = int(model_arg)
            model_id = NUMBER_TO_MODEL_ID.get(num_id)
            if not model_id:
                await self.send_message(
                    room.room_id,
                    f"Unknown model number '{num_id}'. Use !models to see valid IDs."
                )
                return
        else:
            model_id = self.model_aliases.get(model_arg, model_arg)
            if model_id not in MODEL_CLIENT_MAP:
                await self.send_message(room.room_id, f"Unknown model or alias '{model_arg}'. Use !models or !aliases.")
                return

        client_for_model = get_client_for_model_id(model_id)
        if not client_for_model:
            await self.send_message(room.room_id, f"No client found for model '{model_id}'.")
            return

        # Build conversation context from DB
        messages = await self.build_chat_prompt_db(room.room_id)
        messages.append({"role": "user", "content": prompt})

        try:
            resp = await query_model_with_retries(
                client_for_model,
                model_id=model_id,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
            )
            answer = resp.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Error querying model '{model_id}': {e}", exc_info=True)
            await self.send_message(room.room_id, f"Error querying model '{model_id}': {e}")
            return

        # Save the assistant's response in DB for future context
        await save_message(room.room_id, "vLLM", answer, is_bot=True, model_name=model_id)

        # Send the answer to the room
        await self.send_message(room.room_id, f"**{model_id}** says:\n{answer}")

##############################################################################
# EVENT CALLBACK & MAIN
##############################################################################

matrix_bot = MatrixBot(MATRIX_HOMESERVER, MATRIX_USER_ID, MATRIX_PASSWORD)

async def message_callback(room: MatrixRoom, event):
    """matrix-nio callback for new messages."""
    if isinstance(event, RoomMessageText):
        await matrix_bot.on_message(room, event)

matrix_bot.client.add_event_callback(message_callback, RoomMessageText)

def shutdown_handler():
    logger.info("Shutdown signal received; closing Matrix client...")
    asyncio.create_task(matrix_bot.client.close())

async def main():
    # Initialize DB and load model map
    await init_db()
    await initialize_model_map()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, shutdown_handler)

    try:
        await matrix_bot.run()
    except Exception as e:
        logger.exception("Matrix Bot terminated with exception: %s", e)
    finally:
        await matrix_bot.client.close()
        logger.info("Matrix Bot shutdown complete.")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt received, exiting...")
