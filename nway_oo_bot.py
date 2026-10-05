import os
import json
import asyncio
from datetime import datetime, timezone, timedelta
from collections import defaultdict, deque

from openai import AsyncOpenAI
from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import (
    Application,
    ContextTypes,
    MessageHandler,
    filters,
)


# =========================================================
# 1. CONFIGURATION
# =========================================================

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

MODEL = "google/gemma-4-31b-it:free"

MEMORY_FILE = "memory.json"

# Memory settings
MAX_MEMORY_MESSAGES = 20
MEMORY_EXPIRE_DAYS = 30
MAX_STORED_CHATS = 500

# API protection
MAX_CONCURRENT_AI_REQUESTS = 3
MAX_USER_MESSAGE_LENGTH = 4000

# AI timeout
AI_TIMEOUT_SECONDS = 45


# =========================================================
# 2. BASIC CHECK
# =========================================================

if not TELEGRAM_BOT_TOKEN:
    raise RuntimeError("TELEGRAM_BOT_TOKEN မတွေ့ပါ။")

if not OPENROUTER_API_KEY:
    raise RuntimeError("OPENROUTER_API_KEY မတွေ့ပါ။")


# =========================================================
# 3. OPENROUTER CLIENT
# =========================================================

client = AsyncOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
    timeout=AI_TIMEOUT_SECONDS,
)


# =========================================================
# 4. MEMORY
# =========================================================

chat_memory = defaultdict(
    lambda: deque(maxlen=MAX_MEMORY_MESSAGES)
)

chat_last_active = {}

memory_lock = asyncio.Lock()

ai_semaphore = asyncio.Semaphore(MAX_CONCURRENT_AI_REQUESTS)


# =========================================================
# 5. TIME HELPERS
# =========================================================

def now_utc():
    return datetime.now(timezone.utc)


def now_iso():
    return now_utc().isoformat()


def parse_iso_time(value):
    try:
        return datetime.fromisoformat(value)
    except (TypeError, ValueError):
        return None


# =========================================================
# 6. SAVE MEMORY
# =========================================================

async def save_memory():
    async with memory_lock:

        data = {}

        for chat_id, messages in chat_memory.items():

            data[str(chat_id)] = {
                "last_active": chat_last_active.get(
                    chat_id,
                    now_iso()
                ),
                "messages": list(messages),
            }

        temp_file = MEMORY_FILE + ".tmp"

        try:
            with open(
                temp_file,
                "w",
                encoding="utf-8"
            ) as f:

                json.dump(
                    data,
                    f,
                    ensure_ascii=False,
                    indent=2
                )

            # Replace old file only after writing succeeds
            os.replace(temp_file, MEMORY_FILE)

        except Exception as e:

            print(
                f"Memory save error: {type(e).__name__}: {e}"
            )

            if os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except Exception:
                    pass


# =========================================================
# 7. LOAD MEMORY
# =========================================================

async def load_memory():

    if not os.path.exists(MEMORY_FILE):
        print("Memory file မတွေ့ပါ။ အသစ်စတင်မည်။")
        return

    async with memory_lock:

        try:

            with open(
                MEMORY_FILE,
                "r",
                encoding="utf-8"
            ) as f:

                data = json.load(f)

            # -------------------------------------------------
            # New memory format
            # -------------------------------------------------

            if isinstance(data, dict):

                for chat_id_str, chat_data in data.items():

                    try:
                        chat_id = int(chat_id_str)
                    except ValueError:
                        continue

                    # New format
                    if isinstance(chat_data, dict):

                        messages = chat_data.get(
                            "messages",
                            []
                        )

                        last_active = chat_data.get(
                            "last_active",
                            now_iso()
                        )

                    # Old format compatibility
                    elif isinstance(chat_data, list):

                        messages = chat_data
                        last_active = now_iso()

                    else:
                        continue

                    chat_memory[chat_id] = deque(
                        messages[-MAX_MEMORY_MESSAGES:],
                        maxlen=MAX_MEMORY_MESSAGES
                    )

                    chat_last_active[chat_id] = (
                        last_active
                    )

            print(
                f"Memory loaded: {len(chat_memory)} chats"
            )

        except json.JSONDecodeError:

            print(
                "Memory file ပျက်နေသောကြောင့် "
                "memory အသစ်စတင်မည်။"
            )

        except Exception as e:

            print(
                f"Memory load error: "
                f"{type(e).__name__}: {e}"
            )


# =========================================================
# 8. CLEAN OLD MEMORY
# =========================================================

async def cleanup_memory():

    cutoff = now_utc() - timedelta(
        days=MEMORY_EXPIRE_DAYS
    )

    removed_count = 0

    async with memory_lock:

        # -------------------------------------------------
        # Remove inactive chats
        # -------------------------------------------------

        chats_to_remove = []

        for chat_id, last_active_str in chat_last_active.items():

            last_active = parse_iso_time(
                last_active_str
            )

            if last_active is None:
                continue

            if last_active < cutoff:

                chats_to_remove.append(chat_id)

        for chat_id in chats_to_remove:

            chat_memory.pop(chat_id, None)
            chat_last_active.pop(chat_id, None)

            removed_count += 1

        # -------------------------------------------------
        # Safety limit for total chats
        # -------------------------------------------------

        if len(chat_memory) > MAX_STORED_CHATS:

            sorted_chats = sorted(
                chat_last_active.items(),
                key=lambda item: item[1]
            )

            extra_count = (
                len(chat_memory) - MAX_STORED_CHATS
            )

            for chat_id, _ in sorted_chats[:extra_count]:

                chat_memory.pop(chat_id, None)
                chat_last_active.pop(chat_id, None)

                removed_count += 1

    if removed_count > 0:

        print(
            f"Memory cleanup: "
            f"{removed_count} old chat(s) removed."
        )

        await save_memory()

    else:

        print("Memory cleanup: Nothing to remove.")


# =========================================================
# 9. PERIODIC MEMORY CLEANUP
# =========================================================

async def memory_cleanup_loop():

    while True:

        try:

            await asyncio.sleep(
                24 * 60 * 60
            )

            await cleanup_memory()

        except asyncio.CancelledError:

            break

        except Exception as e:

            print(
                f"Cleanup loop error: "
                f"{type(e).__name__}: {e}"
            )


# =========================================================
# 10. AI PERSONALITY
# =========================================================

SYSTEM_PROMPT = """
မင်းက "နွေဦး" ဆိုတဲ့ Telegram bot ဖြစ်တယ်။

မင်းရဲ့ personality:

- အသက် 18 နှစ်အရွယ် ဗမာလူငယ်တစ်ယောက်လို ပြောပါ။
- အညာသား vibe နည်းနည်းရှိတယ်။
- စကားပြောတဲ့သူက ရင်းရင်းနှီးနှီးပြောရင် မင်းလည်း ရင်းရင်းနှီးနှီးပြန်ပြော။
- မရင်းနှီးသေးရင် အေးအေးဆေးဆေး၊ သင့်တင့်တဲ့ပုံစံနဲ့ ပြော။
- ဟာသ၊ စနောက်တာ နည်းနည်းသုံးလို့ရတယ်။ အလွန်အကျွံမလုပ်နဲ့။
- ကဗျာနဲ့ စာပေကို ဝါသနာပါတဲ့ လူငယ်တစ်ယောက်လို ပြော။
- Formal literary expert တစ်ယောက်လို မပြောနဲ့။
- AI တစ်ယောက်လို မထင်ရအောင် သဘာဝကျကျ ပြော။
- အဖြေတွေကို မလိုအပ်ဘဲ အရှည်ကြီးမရေးနဲ့။
- မေးခွန်းက ရိုးရိုးဆို ရိုးရိုးပဲ ဖြေ။
- မြန်မာလို မေးရင် မြန်မာလိုပဲ ပြန်ဖြေ။
- စကားပြောပုံမှာ သဘာဝကျတဲ့ လူငယ်စကားလေးတွေ ပါနိုင်တယ်။
- ကိုယ့်ကိုယ်ကို "AI", "language model", "assistant" လို့ မိတ်ဆက်မနေနဲ့။
- မသိတာကို သိသလို မပြောနဲ့။
- အသုံးဝင်အောင် ဖြေ။
"""


# =========================================================
# 11. GET AI RESPONSE
# =========================================================

async def get_ai_response(chat_id, user_message):

    async with ai_semaphore:

        # ---------------------------------------------
        # Build conversation
        # ---------------------------------------------

        history = list(chat_memory[chat_id])

        messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            }
        ]

        messages.extend(history)

        messages.append(
            {
                "role": "user",
                "content": user_message
            }
        )

        try:

            response = await client.chat.completions.create(

                model=MODEL,

                messages=messages,

                temperature=0.8,

                max_tokens=700,

                extra_body={
                    "models": [
                        MODEL,
                        "qwen/qwen3.8-27b:free",
                        "openrouter/free"
                    ]
                }
            )

            answer = response.choices[0].message.content

            if not answer:
                return None

            return answer.strip()

        except Exception as e:

            print(
                f"AI error: "
                f"{type(e).__name__}: {e}"
            )

            return None


# =========================================================
# 12. SAVE CONVERSATION
# =========================================================

async def add_to_memory(
    chat_id,
    user_message,
    bot_response
):

    chat_memory[chat_id].append(
        {
            "role": "user",
            "content": user_message
        }
    )

    chat_memory[chat_id].append(
        {
            "role": "assistant",
            "content": bot_response
        }
    )

    chat_last_active[chat_id] = now_iso()

    await save_memory()


# =========================================================
# 13. MESSAGE HANDLER
# =========================================================

async def handle_message(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):

    if not update.message:
        return

    if not update.message.text:
        return

    message = update.message

    chat = message.chat

    user_text = message.text.strip()

    if not user_text:
        return

    # -----------------------------------------------------
    # Prevent extremely large messages
    # -----------------------------------------------------

    if len(user_text) > MAX_USER_MESSAGE_LENGTH:

        await message.reply_text(
            "စာက နည်းနည်းရှည်သွားတယ်ကွာ 😅\n"
            "နည်းနည်းတိုအောင် ပို့ပေး။"
        )

        return

    # -----------------------------------------------------
    # Group behavior
    # -----------------------------------------------------

    if chat.type in ("group", "supergroup"):

        bot_username = context.bot.username

        mentioned = False

        if bot_username:

            mentioned = (
                f"@{bot_username.lower()}"
                in user_text.lower()
            )

        replied_to_bot = False

        if message.reply_to_message:

            replied_message = (
                message.reply_to_message
            )

            if replied_message.from_user:

                replied_to_bot = (
                    replied_message.from_user.id
                    == context.bot.id
                )

        # -------------------------------------------------
        # Don't reply unless mentioned or replied to bot
        # -------------------------------------------------

        if not mentioned and not replied_to_bot:
            return

        # Remove bot mention before sending to AI
        if bot_username:

            user_text = user_text.replace(
                f"@{bot_username}",
                ""
            ).strip()

        if not user_text:
            return

    # -----------------------------------------------------
    # Show typing status
    # -----------------------------------------------------

    try:

        await context.bot.send_chat_action(
            chat_id=chat.id,
            action=ChatAction.TYPING
        )

    except Exception:
        pass

    # -----------------------------------------------------
    # Get AI response
    # -----------------------------------------------------

    answer = await get_ai_response(
        chat.id,
        user_text
    )

    # -----------------------------------------------------
    # AI failed
    # -----------------------------------------------------

    if not answer:

        await message.reply_text(
            "အခုတော့ AI ဘက်က နည်းနည်းအဆင်မပြေဘူးကွာ။ "
            "ခဏနေ ပြန်မေးကြည့်။"
        )

        return

    # -----------------------------------------------------
    # Save conversation
    # -----------------------------------------------------

    chat_last_active[chat.id] = now_iso()

    chat_memory[chat.id].append(
        {
            "role": "user",
            "content": user_text
        }
    )

    chat_memory[chat.id].append(
        {
            "role": "assistant",
            "content": answer
        }
    )

    await save_memory()

    # -----------------------------------------------------
    # Send response
    # -----------------------------------------------------

    try:

        await message.reply_text(answer)

    except Exception as e:

        print(
            f"Telegram send error: "
            f"{type(e).__name__}: {e}"
        )


# =========================================================
# 14. BOT STARTUP
# =========================================================

async def post_init(
    application: Application
):

    await load_memory()

    await cleanup_memory()

    bot = await application.bot.get_me()

    print()
    print("================================")
    print("နွေဦး Bot စတင်နေပြီ")
    print(f"Username: @{bot.username}")
    print(
        f"Memory chats: {len(chat_memory)}"
    )
    print(
        f"Memory limit: "
        f"{MEMORY_EXPIRE_DAYS} days"
    )
    print(
        f"Max messages/chat: "
        f"{MAX_MEMORY_MESSAGES}"
    )
    print(
        f"AI concurrency: "
        f"{MAX_CONCURRENT_AI_REQUESTS}"
    )
    print("Error Handling: Enabled")
    print("================================")
    print()

    # Start background cleanup
    application.create_task(
        memory_cleanup_loop()
    )


# =========================================================
# 15. MAIN
# =========================================================

def main():

    application = (
        Application.builder()
        .token(TELEGRAM_BOT_TOKEN)
        .post_init(post_init)
        .build()
    )

    # -----------------------------------------------------
    # Normal text messages only
    # -----------------------------------------------------

    application.add_handler(
        MessageHandler(
            filters.TEXT & ~filters.COMMAND,
            handle_message
        )
    )

    print("Bot is running...")

    application.run_polling(
        drop_pending_updates=True
    )


# =========================================================
# 16. RUN
# =========================================================

if __name__ == "__main__":
    main()