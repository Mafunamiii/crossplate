import logging

class Color:
    RESET = "\033[0m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"

class ColoredFormatter(logging.Formatter):
    def format(self, record):
        prefix = f"{Color.BLUE}[LOGGER] {Color.RESET}"

        original_msg = record.msg

        if record.levelno == logging.DEBUG:
            record.msg = f"{Color.CYAN}{original_msg}{Color.RESET}"
        elif record.levelno == logging.INFO:
            record.msg = f"{Color.GREEN}{original_msg}{Color.RESET}"
        elif record.levelno == logging.WARNING:
            record.msg = f"{Color.YELLOW}{original_msg}{Color.RESET}"
        elif record.levelno == logging.ERROR:
            record.msg = f"{Color.RED}{original_msg}{Color.RESET}"
        elif record.levelno == logging.CRITICAL:
            record.msg = f"{Color.MAGENTA}{original_msg}{Color.RESET}"

        formatted_message = super().format(record)
        return f"{prefix}{formatted_message}"

def setup_logger(name="Day 4", level=logging.DEBUG):
    logger = logging.getLogger(name)
    logger.setLevel(level)

    logger.propagate = False

    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = ColoredFormatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logger.addHandler(console_handler)

    return logger