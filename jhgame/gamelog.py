import logging


def get_module_logger(modname):
    """Get custom logger.

    Args:
        modname (str): Module name.

    Returns:
        logging.Logger: Custom logger.
    """
    logger = logging.getLogger(modname)
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        fmt="[%(asctime)s] %(levelname)s " + "[%(filename)s %(funcName)s]: %(message)s", datefmt="%y/%m/%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger
