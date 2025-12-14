import logging
import sys

# === ANSI Color Codes ===
RED = '\033[91m'
GREEN = '\033[92m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
ORANGE = '\033[38;5;208m'
PINK = '\033[38;5;205m'
CYAN = '\033[96m'
GRAY = '\033[90m'
RESET = '\033[0m'
BOLD = '\033[1m'


# === Global Mappings ===
LEVEL_COLORS = { # A dictionary mapping log levels to their corresponding ANSI color codes
    'DEBUG': BLUE,
    'INFO': GREEN,
    'WARNING': YELLOW,
    'ERROR': RED,
}

STATUS_COLORS = { # A dictionary mapping status values to their corresponding ANSI color codes
    'success': GREEN,
    'in progress': YELLOW,
    'failed': RED,
    'fallback': ORANGE,
    'skipped': PINK
}

_LEVELS = { # A dictionary mapping human-readable level names (strings) to Python logging constants (integers) 
    "ERROR": logging.ERROR,
    "WARNING": logging.WARNING,
    "INFO": logging.INFO,
    "DEBUG": logging.DEBUG,
}


# === Custom Formatter Class ===
class CustomFormatter(logging.Formatter):
    """Class for custom logging formatter to handle action, status, and data fields in log records;"""
    """Formatters control how log records are turned into strings"""
    def __init__(self):
        super().__init__() # Call parent constructor to initialize base Formatter class

    def format(self, record): # Override format method from Formatter to add custom fields if missing
        if not hasattr(record, 'action'): # Check if record has 'action' attribute
            record.action = '' # If attribute is not provided (e.g., in basic logging calls) so formatting does not crash
        if not hasattr(record, 'status'):
            record.status = ''
        if not hasattr(record, 'data'):
            record.data = ''
        if isinstance(record.data, dict):
            record.data = ", ".join(f"{key}: {value}" for key, value in record.data.items())
        
        level_color = LEVEL_COLORS.get(record.levelname, '') # Get color for log level; Default to no color if level not found
        status_color = STATUS_COLORS.get(record.status.lower(), '') if record.status else '' # Get color for status; Default to no color if status not found or empty
        data_color = CYAN if record.data else '' # Use cyan color for data; Default to no color if data is empty

        time = f"{self.formatTime(record, '%H:%M:%S')}.{int(record.msecs):03d}" # Format time as HH:MM:SS.mmm; int(record.msecs):03d ensures milliseconds are intigers with 3 digits
        
        levelname = record.levelname
        separator = " " * max(0, 8 - len(levelname))
        levelname_display = f"{level_color}{levelname}{RESET}:{separator}"

        if record.action: # Choose format based on whether action is provided (custom format - own logs) or not (default format - external logs)
              name = record.name
              name_display = f"{BOLD}{name}{RESET}"
              action = record.action
              action_display = f"{BOLD}{action}{RESET}"
              status = record.status
              status_display = f"{status_color}{BOLD}status: {RESET}{status_color}{status}{RESET}"
              data = record.data
              data_display = f"{data_color}{BOLD}data: {RESET}{data_color}({data}){RESET}"
              formatted = f"{levelname_display} {name_display} | {action_display} [{status_display}] | [{data_display}] | {time}"
        else:
            message = record.getMessage()
            formatted = f"{levelname_display} {message} | {time}"

        return formatted


# === Custom Adapter Class ===
class CustomAdapter(logging.LoggerAdapter):
    """Class for custom logger adapter to handle action, status, and data as direct keyword arguments"""
    """Adapters allow modification of log records before they are emitted; They sit between the logger and the formatter"""
    def process(self, msg, kwargs): # Override process method from LoggerAdapter to extract action, status and data from kwargs
        extra = kwargs.get('extra', {}) # Check if 'extra' dictionary has been provided in kwargs; if not, create an empty one
        for key in ['action', 'status', 'data']: # Iterate over expected keys
            if key in kwargs:
                extra[key] = kwargs.pop(key) # Delete key from kwargs and add it to extra dictionary
        kwargs['extra'] = extra # Update kwargs with modified extra dictionary
        return msg, kwargs # Return unchanged msg and modified kwargs to the logging system

    def debug(self, msg=None, *args, **kwargs): # Override logging methods to make msg argument optional
        if msg is None:
            msg = ""
        self.log(logging.DEBUG, msg, *args, **kwargs) # After handling msg, call process method internally to modify kwargs

    def info(self, msg=None, *args, **kwargs):
        if msg is None:
            msg = ""
        self.log(logging.INFO, msg, *args, **kwargs)

    def warning(self, msg=None, *args, **kwargs):
        if msg is None:
            msg = ""
        self.log(logging.WARNING, msg, *args, **kwargs)

    def error(self, msg=None, *args, **kwargs):
        if msg is None:
            msg = ""
        self.log(logging.ERROR, msg, *args, **kwargs)


# === Setup Logging Function ===
def setup_logging(level: int | str = "INFO") -> None:
    """Function to setup logging configuration with specified level and format"""
    if isinstance(level, str): # Check if level is a string
        level = _LEVELS.get(level.upper(), logging.INFO) # Convert level string to logging level constant; Default to INFO if invalid level string

    root = logging.getLogger() # Get root logger
    if root.handlers: # Check if root logger already has handlers
        root.setLevel(level) # Set level if already configured
        return # Exit if already configured

    handler = logging.StreamHandler(stream=sys.stdout) # Create stream handler for stdout; Stream log output directly to console
    handler.setFormatter(CustomFormatter()) # Set formatter for handler; CustomFormatter handles both custom and default log formats

    root.setLevel(level) # Set root logger level to capture all messages at this level and above
    root.addHandler(handler) # Add handler to root logger to direct log messages to the console

    logging.getLogger("httpx").setLevel(logging.WARNING) # Hide httpx logs unless they are warnings or errors
    logging.getLogger("httpcore").setLevel(logging.WARNING) # Hide httpcore logs unless they are warnings or errors

    
# === Get Logger Function ===
def get_logger(name: str | None = None) -> logging.LoggerAdapter:
    """Function to get a module-specific logger with custom adapter"""
    logger = logging.getLogger(name if name else __name__) # Name identifies the module requesting the logger; Default to current module name if None
    return CustomAdapter(logger, extra={}) # CustomAdapter inherits from logging.LoggerAdapter which requires two arguments: the logger instance and an initial 'extra' dictionary