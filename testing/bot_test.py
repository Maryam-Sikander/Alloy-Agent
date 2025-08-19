import discord
import asyncio
import os
from dotenv import load_dotenv
from config import Config

config = Config()
load_dotenv()
DISCORD_TOKEN = os.getenv("DISCORD_BOT_TOKEN")

intents = discord.Intents.default()
intents.messages = True
intents.message_content = True

client = discord.Client(intents=intents)

@client.event
async def on_ready():
    print(f"Logged in as {client.user}")
    # Replace CHANNEL_ID with a real channel ID to test
    channel = client.get_channel(config.get('configurable', 'channel-id'))
    if channel:
        await channel.send("Test message: Hello from testing script!")

client.run(DISCORD_TOKEN)
