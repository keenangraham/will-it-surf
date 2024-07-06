import logging

import os

level = os.environ.get('LOG_LEVEL', 'INFO')

logging.basicConfig(level=level, format='%(asctime)s - %(levelname)s - %(message)s')
