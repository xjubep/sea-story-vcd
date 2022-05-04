from datetime import datetime, timedelta, timezone
import logging
import sys
import os

from tensorboardX import SummaryWriter

KST = timezone(timedelta(hours=9))


def initialize_log(path):
    def convert_kst(sec, what):
        return datetime.now(tz=KST).timetuple()

    logging.Formatter.converter = lambda *x: datetime.now(KST).timetuple()
    logging.basicConfig(
        format='[%(asctime)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        level=logging.INFO,
        handlers=[
            logging.FileHandler(filename=path, mode='w'),
            logging.StreamHandler(stream=sys.stdout)
        ])

    logger = logging.getLogger()

    return logger


def initialize_writer_and_log(log_dir, comment=None):
    current = datetime.now(tz=KST)
    current_date = current.strftime('%Y%m%d')
    current_time = current.strftime('%H%M%S')

    if comment is None or comment == '':
        comment = current_time

    log_dir = os.path.join(log_dir, current_date, comment)
    if os.path.exists(log_dir):
        log_dir += f'_{current_time}'

    os.makedirs(log_dir)
    logger = initialize_log(os.path.join(log_dir, 'log.txt'))
    writer = SummaryWriter(logdir=log_dir)

    import socket
    logger.info("=========================================================")
    logger.info(f'Start - {socket.gethostname()}')
    logger.info(f'Log directory ... {log_dir}')
    logger.info("=========================================================")

    return writer, logger, log_dir
