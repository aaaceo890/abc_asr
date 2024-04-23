import logging
import time
import os.path as path


def get_logger(
        verbose: int = 1,
        log_out_dir: str = None,
        log_format: str = '%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s'
):
    logger = logging.getLogger()
    if verbose == 0:
        logger.setLevel(logging.DEBUG)
    elif verbose == 1:
        logger.setLevel(logging.INFO)
    elif verbose == 2:
        logger.setLevel(logging.WARNING)
    elif verbose == 3:
        logger.setLevel(logging.ERROR)
    elif verbose == 4:
        logger.setLevel(logging.CRITICAL)

    formatter = logging.Formatter(log_format)

    str_han = logging.StreamHandler()
    str_han.setFormatter(formatter)
    logger.addHandler(str_han)

    if log_out_dir is not None:
        rq = time.strftime('%Y-%m-%d-%H:%M', time.localtime(time.time()))
        logname = rq + '.log'
        logfile = path.join(log_out_dir, logname)
        file_han = logging.FileHandler(logfile, mode='w')
        file_han.setFormatter(formatter)
        logger.addHandler(file_han)

    return logger
