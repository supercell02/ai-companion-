import asyncio
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from config import config
from core.chat_loop import ChatLoop
from tui.app import CompanionApp

async def main():
    chat_loop = ChatLoop(config)
    await chat_loop.initialize()
    app = CompanionApp(chat_loop=chat_loop)
    await app.run_async()

if __name__ == "__main__":
    asyncio.run(main())