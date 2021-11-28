import os
import logging

logger = logging.getLogger(__name__)
name = os.environ.get("NAME", "empty")
print(name)
logger = logger.error(name)