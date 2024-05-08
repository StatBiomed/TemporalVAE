"""
Code modified from:
https://github.com/ignavier/golem/blob/main/src/utils/logger.py
"""
from datetime import datetime
import logging
import platform
import subprocess
import sys

import psutil
from pytz import timezone, utc
import torch

from datetime import datetime
import logging
import sys

from pytz import timezone, utc

from datetime import datetime
import logging
import os
import pathlib

from pytz import timezone


_logger = logging.getLogger(__name__)
def gpu_info() -> str:
    info = ''
    for id in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(id)
        info += f'cuda:{id} ({p.name}, {p.total_memory / (1 << 20):.0f}MiB)\n'
    return info[:-1]


def setup_logger(log_path, level='INFO'):
    """Set up logger.

    Args:
        log_path (str): Path to create the log file.
        level (str): Logging level. Default: 'INFO'.
    """
    log_format = '%(asctime)s %(levelname)s - %(name)s - %(message)s'

    def custom_time(*args):
        utc_dt = utc.localize(datetime.utcnow())
        my_tz = timezone('EST')
        converted = utc_dt.astimezone(my_tz)
        return converted.timetuple()

    logging.basicConfig(
        filename=log_path,
        level=logging.getLevelName(level),
        format=log_format,
    )

    logging.Formatter.converter = custom_time

    # Set up logging to console
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    console.setFormatter(logging.Formatter(log_format))
    # Add the console handler to the root logger
    logging.getLogger('').addHandler(console)

    # Log for unhandled exception
    logger = logging.getLogger(__name__)
    sys.excepthook = lambda *ex: logger.critical("Unhandled exception.", exc_info=ex)


def get_system_info():
    # Code modified from https://stackoverflow.com/a/58420504
    try:
        info = {}
        info['git_revision_hash'] = get_git_revision_hash()
        info['platform'] = platform.system()
        info['platform-release'] = platform.release()
        info['platform-version'] = platform.version()
        info['architecture'] = platform.machine()
        info['processor'] = platform.processor()
        info['ram'] = '{} GB'.format(round(psutil.virtual_memory().total / (1024.0 ** 3)))
        info['cpu_count'] = psutil.cpu_count()
        info['cpu_count'] = psutil.cpu_count()

        # Calculate percentage of available memory
        # Referred from https://stackoverflow.com/a/2468983
        info['percent_available_ram'] = psutil.virtual_memory().available * 100 / psutil.virtual_memory().total
        return info
    except Exception as e:
        return None


def get_git_revision_hash():
    # Referred from https://stackoverflow.com/a/21901260
    try:
        return str(subprocess.check_output(['git', 'rev-parse', 'HEAD']).strip())[2:-1]
    except:
        return ''
def create_dir(output_dir):
    """Create directory.

    Args:
        output_dir (str): A directory to create if not found.

    Returns:
        exit_code: 0 (success) or -1 (failed).
    """
    try:
        if not os.path.exists(output_dir):
            pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

        return 0
    except Exception as err:
        _logger.critical("Error when creating directory: {}.".format(err))
        exit(-1)


def get_datetime_str():
    """Get string based on current datetime."""
    return datetime.now(timezone('EST')).strftime('%Y-%m-%d_%H-%M-%S-%f')[:-3]

class LogHelper:
    """Helper class to set up logger."""
    log_format = '%(asctime)s %(levelname)s - %(name)s - %(message)s '

    @staticmethod
    def setup(log_path, level='INFO'):
        """Set up logger.

        Args:
            log_path (str): Path to create the log file.
            level (str): Logging level. Default: 'INFO'.
        """

        def custom_time(*args):
            utc_dt = utc.localize(datetime.utcnow())
            my_tz = timezone('EST')
            converted = utc_dt.astimezone(my_tz)
            return converted.timetuple()

        logging.basicConfig(
            filename=log_path,
            level=logging.getLevelName(level),
            format=LogHelper.log_format,
        )

        logging.Formatter.converter = custom_time

        # Set up logging to console
        console = logging.StreamHandler()
        console.setLevel(logging.DEBUG)
        console.setFormatter(logging.Formatter(LogHelper.log_format))
        # Add the console handler to the root logger
        logging.getLogger('').addHandler(console)

        # Log for unhandled exception
        logger = logging.getLogger(__name__)
        sys.excepthook = lambda *ex: logger.critical("Unhandled exception.", exc_info=ex)
