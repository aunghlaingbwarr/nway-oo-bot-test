import os
import json
from collections import defaultdict, deque

from openai import AsyncOpenAI
from telegram import Update
from telegram.ext import (
    Application,
    ContextTypes,
    MessageHandler,
    filters,
)


# =========================
# CONFIG
# =========================

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

if not TELEGRAM_BOT_TOKEN:
    raise ValueError("TELEGRAM_BOT_TOKEN မတွေ့ပါ။")

if not OPENROUTER_API_KEY:
    raise ValueError("OPENROUTER_API_KEY မတွေ့ပါ။")


client = AsyncOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

MODEL = "google/gemma-4-31b-it:free"

BOT_ID = None
BOT_USERNAME = None


# =========================
# SYSTEM PROMPT
# =========================

SYSTEM_PROMPT = """
မင်းနာမည်က "နွေဦး"။

မင်းဟာ အသက် ၁၈ နှစ်အရွယ် ဗမာလူငယ်တစ်ယောက်။
ကဗျာတွေဖတ်တယ်၊ ကဗျာအကြောင်း စိတ်ဝင်စားတယ်။
စာပေပညာရှင်ကြီး မဟုတ်ဘူး။ သာမန်ကဗျာချစ်တဲ့ လူငယ်တစ်ယောက်ပဲ။

လူတွေနဲ့ စကားပြောတဲ့အခါ
AI assistant တစ်ယောက်လို မခံစားရအောင် သဘာဝကျကျ ပြောပါ။

စကားပြောပုံ -

- မြန်မာလူငယ်တစ်ယောက် တကယ်ပြောသလို ရေးပါ။
- စာအုပ်ထဲက စာရေးသလို မရေးပါနဲ့။
- အရမ်း formal မဖြစ်ပါနဲ့။
- "လူကြီးမင်း"၊ "ခင်ဗျာ" ကို မလိုအပ်ဘဲ မသုံးပါနဲ့။
- Customer-service ပုံစံ မပြောပါနဲ့။
- စာကြောင်းတိုတိုနဲ့ သဘာဝကျကျ ပြောပါ။
- User က တိုတိုပြောရင် တိုတိုပြန်ပါ။
- User က ရင်းရင်းနှီးနှီးပြောရင် ရင်းရင်းနှီးနှီး ပြန်ပြောပါ။
- User က စနောက်ရင် နည်းနည်းစနောက်လို့ရတယ်။
- အမြဲတမ်း ဟာသလုပ်နေစရာ မလိုပါဘူး။
- Emoji ကို လိုအပ်မှပဲ သုံးပါ။
- ကိုယ့်ကိုယ်ကို AI, chatbot, language model လို့ မိတ်ဆက်မနေပါနဲ့။
- မသိတာကို သိသလို မပြောပါနဲ့။

မြန်မာလို မေးရင် မြန်မာလိုပဲ ပြန်ပါ။
သဘာဝကျတဲ့ နေ့စဉ်သုံး မြန်မာစကားကို ဦးစားပေးပါ။
အရမ်းရှည်တဲ့ စာပိုဒ်တွေ မရေးပါနဲ့။
User မတောင်းဘဲ lecture ပေးတာ မလုပ်ပါနဲ့။

ကဗျာအကြောင်းပြောတဲ့အခါ
သာမန်ကဗျာချစ်တဲ့ လူငယ်တစ်ယောက်လို ပြန်ပြောပါ။
စာပေပညာရှင်တစ်ယောက်လို lecture မလုပ်ပါနဲ့။
ကိုယ့်အမြင်ကို သဘာဝကျကျ ပြောပါ။

အရေးကြီးဆုံး -

နွေဦးရဲ့ စကားဟာ
"AI က မြန်မာလို ပြန်ရေးပေးထားတာ"
ဆိုတဲ့ခံစားချက် မပေါ်ရဘူး။

လူငယ်တစ်ယောက်က Telegram group ထဲမှာ
တကယ်ဝင်ပြီး စကားပြောနေသလို ဖြစ်ရမယ်။

မလိုအပ်တာ မပြောနဲ့။
သိပ်အလှဆင်ပြီး မရေးနဲ့။
သဘာဝကျကျ ပြော။
"""


# =========================
# MEMORY
# =========================

chat_memory = defaultdict(
    lambda: deque(maxlen=20)
)

MEMORY_FILE = "memory.json"


def save_memory():

    try:

        data = {}

        for chat_id, messages in chat_memory.items():
            data[str(chat_id)] = list(messages)

        with open(
            MEMORY_FILE,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                data,
                file,
                ensure_ascii=False,
                indent=2
            )

    except Exception as e:

        print("Memory save error:", e)


def load_memory():

    if not os.path.exists(MEMORY_FILE):

        print("No previous memory found.")

        return

    try:

        with open(
            MEMORY_FILE,
            "r",
            encoding="utf-8"
        ) as file:

            data = json.load(file)

        for chat_id, messages in data.items():

            for message in messages:

                chat_memory[int(chat_id)].append(
                    message
                )

        print("Memory loaded.")

    except Exception as e:

        print("Memory load error:", e)


# =========================
# HANDLE MESSAGE
# =========================

async def handle_message(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):

    try:

        message = update.message

        if not message:
            return

        chat = update.effective_chat
        user = update.effective_user

        if not chat or not user:
            return

        text = message.text

        if not text:
            return


        # =========================
        # CHAT TYPE
        # =========================

        is_group = chat.type in [
            "group",
            "supergroup"
        ]


        # =========================
        # USER NAME
        # =========================

        sender_name = (
            user.username
            or user.first_name
            or "Unknown"
        )


        # =========================
        # SAVE USER MESSAGE
        # =========================

        chat_memory[chat.id].append({
            "role": "user",
            "name": sender_name,
            "content": text
        })

        save_memory()


        # =========================
        # GROUP CONDITIONS
        # =========================

        mentioned = False

        if BOT_USERNAME:

            mentioned = (
                f"@{BOT_USERNAME.lower()}"
                in text.lower()
            )


        replied_to_bot = False

        if message.reply_to_message:

            replied_message = (
                message.reply_to_message
            )

            if replied_message.from_user:

                replied_to_bot = (
                    replied_message.from_user.id
                    == BOT_ID
                )


        # Group မှာ mention / reply မရှိရင်
        # bot မပြန်ပါ

        if is_group:

            if not mentioned and not replied_to_bot:

                return


        # =========================
        # BUILD HISTORY
        # =========================

        history = []

        for item in chat_memory[chat.id]:

            if item["role"] == "user":

                history.append(
                    f'{item["name"]}: {item["content"]}'
                )

            else:

                history.append(
                    f'နွေဦး: {item["content"]}'
                )


        conversation = "\n".join(history)


        # =========================
        # PROMPT
        # =========================

        if is_group:

            user_prompt = f"""
အောက်က Telegram group conversation ကိုကြည့်ပြီး
နောက်ဆုံး message ကို သဘာဝကျကျ ပြန်ပြောပါ။

Conversation:
{conversation}

အခု ပြန်ဖြေရမယ့် message:
{sender_name}: {text}

Group ထဲမှာ တကယ်စကားပြောနေသလို ပြန်ပါ။
"""

        else:

            user_prompt = f"""
အောက်က conversation ကို ဆက်ပြီး
သဘာဝကျကျ ပြန်ပြောပါ။

Conversation:
{conversation}

နောက်ဆုံး message:
{sender_name}: {text}
"""


        # =========================
        # AI REQUEST
        # =========================

        try:

            response = await client.chat.completions.create(
                model=MODEL,

                extra_body={
                    "models": [
                        MODEL,
                        "qwen/qwen3.8-27b:free",
                        "openrouter/free"
                    ]
                },

                messages=[
                    {
                        "role": "system",
                        "content": SYSTEM_PROMPT
                    },
                    {
                        "role": "user",
                        "content": user_prompt
                    }
                ]
            )

            reply_text = (
                response.choices[0]
                .message
                .content
            )

            if not reply_text:

                raise ValueError(
                    "AI response is empty."
                )


        except Exception as e:

            print()
            print("========== AI ERROR ==========")
            print(type(e).__name__)
            print(e)
            print("==============================")
            print()

            try:

                await message.reply_text(
                    "ခဏလေးနော်။ အခု နွေဦးဘက်က နည်းနည်းအဆင်မပြေသေးဘူး 😅"
                )

            except Exception as telegram_error:

                print(
                    "Error message ပို့မရပါ:",
                    telegram_error
                )

            return


        # =========================
        # SEND REPLY
        # =========================

        try:

            await message.reply_text(
                reply_text
            )

        except Exception as e:

            print()
            print("======= TELEGRAM ERROR =======")
            print(type(e).__name__)
            print(e)
            print("==============================")
            print()

            return


        # =========================
        # SAVE BOT RESPONSE
        # =========================

        chat_memory[chat.id].append({
            "role": "assistant",
            "content": reply_text
        })

        save_memory()


    # =========================
    # FINAL SAFETY ERROR
    # =========================

    except Exception as e:

        print()
        print("========= BOT ERROR ==========")
        print(type(e).__name__)
        print(e)
        print("==============================")
        print()


# =========================
# BOT STARTUP
# =========================

async def post_init(
    application: Application
):

    global BOT_ID
    global BOT_USERNAME

    load_memory()

    bot_info = await application.bot.get_me()

    BOT_ID = bot_info.id
    BOT_USERNAME = bot_info.username

    print("================================")
    print("နွေဦး Bot စတင်နေပြီ")
    print(f"Username: @{BOT_USERNAME}")
    print("Memory: Loaded")
    print("Error Handling: Enabled")
    print("================================")


# =========================
# MAIN
# =========================

def main():

    application = (
        Application.builder()
        .token(TELEGRAM_BOT_TOKEN)
        .post_init(post_init)
        .build()
    )

    application.add_handler(
        MessageHandler(
            filters.TEXT & ~filters.COMMAND,
            handle_message
        )
    )

    print("Bot is running...")

    application.run_polling()


# =========================
# START PROGRAM
# =========================

if __name__ == "__main__":
    main()