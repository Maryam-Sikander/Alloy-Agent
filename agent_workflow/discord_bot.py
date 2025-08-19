import discord
import os
import re
import asyncio
import logging
import time
from dotenv import load_dotenv
from agent_workflow.orchestrator import init_orchestrator
from agent_workflow.calendar_workers import calendar_worker_summary_list

# -------------------- Logging --------------------
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.DEBUG
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# -------------------- Discord Client --------------------
intents = discord.Intents.default()
intents.messages = True
intents.message_content = True  # Required to read message content

bot = discord.Client(intents=intents)

# -------------------- Markdown Helpers --------------------
def apply_markdown_replacements(text):
    replacements = {
        "+": "\\+",
        "#": "\\#",
        "-": "\\-",
        "=": "\\=",
        "{{": "\\{{",
        "}}": "\\}}",
        ".": "\\.",
        "!": "\\!",
        "~": "\\~",
        "`": "\\`",
        ">": "\\>",
        "|": "\\|",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    text = re.sub(r"(?<!\*)\*(?!\*)", r"\*", text)
    text = re.sub(r"(?<!\_)\_(?!\_)", r"\_", text)
    text = re.sub(r"(\*{3,})", lambda m: "\\" + "\\".join(m.group()), text)
    text = re.sub(r"(\_{3,})", lambda m: "\\" + "\\".join(m.group()), text)
    text = re.sub(r"\*\*(\S.*?\S)\*\*", r"*\1*", text)
    text = re.sub(r"\_\_(\S.*?\S)\_\_", r"_\1_", text)
    text = re.sub(r"\*\*", r"\*\*", text)
    text = re.sub(r"\_\_", r"\_\_", text)
    return text

def escape_special_characters(text):
    markdown_link_pattern = r"\[[^\]]+\]\([^)]*\)"
    segments = re.split(f"({markdown_link_pattern})", text)
    special_chars_pattern = r"([\[\]\(\)])"
    for i, segment in enumerate(segments):
        if re.fullmatch(markdown_link_pattern, segment):
            continue
        segments[i] = re.sub(special_chars_pattern, r"\\\1", segment)
    return "".join(segments)

# -------------------- Help Message --------------------
help_message = (
    "You can ask me anything, and I'll try to respond using AI. "
    "However, my main focus is managing the following calendars: "
    f"{', '.join(calendar_worker_summary_list)}. "
    "If your question is related to scheduling, events, or availability within these calendars, I will provide accurate information. "
    "For other topics, I may not always have the answer, but I'll do my best to assist you or guide you accordingly."
)

# -------------------- Typing Simulation --------------------
async def send_typing_action(channel):
    while True:
        await channel.trigger_typing()
        await asyncio.sleep(4.5)

# -------------------- Event Handlers --------------------
@bot.event
async def on_ready():
    logger.info(f"Bot logged in as {bot.user}")

@bot.event
async def on_message(message):
    if message.author == bot.user:
        return

    # Handle commands
    if message.content.startswith("/start"):
        await message.channel.send(f"Hello {message.author.mention}! {help_message}")
        return

    if message.content.startswith("/help"):
        await message.channel.send(help_message)
        return

    # Handle normal messages
    typing_task = asyncio.create_task(send_typing_action(message.channel))
    try:
        config = {
            "configurable": {
                "thread_id": message.author.name or str(message.author.id) or "unknown_user"
            }
        }
        logger.debug(f"Invoking orchestrator_graph with config: {config}")
        init_time = time.time()
        orchestrator_graph = await init_orchestrator()
        response = await orchestrator_graph.ainvoke(
            {"user_input": message.content},
            config,
        )
        text =  response["messages"][-1].content
        duration = time.time() - init_time
        logger.debug(f"Response generated successfully: {duration:.4f}")

        try:
            await message.channel.send(
                escape_special_characters(apply_markdown_replacements(text))
            )
        except Exception as parse_exc:
            logger.warning(f"Failed to send escaped message: {parse_exc}")
            await message.channel.send(text)

    except Exception as e:
        logger.error(f"Error processing message: {e}", exc_info=True)
        await message.channel.send(
            f"Sorry, I couldn't process your request right now.\nError: {e}"
        )
    finally:
        typing_task.cancel()

# -------------------- Run Bot --------------------
if __name__ == "__main__":
    load_dotenv()
    DISCORD_TOKEN = os.getenv("DISCORD_BOT_TOKEN")

    if not DISCORD_TOKEN:
        logger.error("DISCORD_BOT_TOKEN environment variable not found.")
    else:
        bot.run(DISCORD_TOKEN)
