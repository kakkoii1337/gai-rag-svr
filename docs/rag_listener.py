# prettier-ignore
import asyncio
from gai.lib.common.logging import getLogger
logger = getLogger(__name__)
from gai.lib.common.StatusListener import StatusListener

async def callback(message):
    logger.info(f"Callback: message={message}")

listener = StatusListener("ws://localhost:12036/gen/v1/rag/index-file/ws","12345")
asyncio.run(listener.listen(callback))