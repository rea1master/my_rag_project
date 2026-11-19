import logging
from set_log import set_logger



set_logger()

logger = logging.getLogger(__name__)
logger.error("是你想的这样吗")
print(f"[main.py] 我拿到的 logger.name = {logger.name}")
