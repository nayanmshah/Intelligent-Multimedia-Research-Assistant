import os
import logging

def setup_logger(name="IMRA", level=logging.INFO):
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(level)
    return logger

def load_env_variables(dotenv_path=".env"):
    from dotenv import load_dotenv
    load_dotenv(dotenv_path)
    return os.environ