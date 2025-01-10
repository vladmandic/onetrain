import functools
import logging
import logging.handlers
import warnings
import platform
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import Progress, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn, TimeElapsedColumn
from tqdm import tqdm


console = Console(log_time=True, log_time_format='%H:%M:%S-%f')
log = logging.getLogger(__name__)
pbar = None


class HostnameFilter(logging.Filter):
    hostname = platform.node()

    def filter(self, record):
        record.hostname = HostnameFilter.hostname
        return True


def init_logger(log_file: str = None):
    global pbar # pylint: disable=global-statement
    logging.basicConfig(level=logging.DEBUG, handlers=[logging.NullHandler()])
    warnings.filterwarnings(action="ignore", category=DeprecationWarning)
    warnings.filterwarnings(action="ignore", category=FutureWarning)
    warnings.filterwarnings(action="ignore", category=UserWarning)
    tqdm.__init__ = functools.partialmethod(tqdm.__init__, disable=True) # hide onetrainer tqdm progress bar
    pbar = Progress(
        TextColumn('[cyan]{task.description}'),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        TextColumn('[cyan]{task.fields[text]}'),
        console=console,
        transient=False)

    # log console handler
    log_handler_console = RichHandler(show_time=True, omit_repeated_times=False, show_level=True, show_path=False, markup=False, rich_tracebacks=True, log_time_format='%H:%M:%S-%f', level=logging.DEBUG, console=console)
    log_handler_console.setLevel(logging.DEBUG)
    log.addHandler(log_handler_console)

    # log to file
    if log_file is not None:
        log_handler_file = logging.handlers.RotatingFileHandler(log_file, encoding='utf-8', delay=True)
        log_handler_file.addFilter(HostnameFilter())
        log_handler_file.formatter = logging.Formatter('%(asctime)s | %(hostname)s | %(levelname)s | %(name)s | %(module)s | %(message)s')
        log_handler_file.setLevel(logging.DEBUG)
        log.addHandler(log_handler_file)
    log.setLevel(logging.INFO)
