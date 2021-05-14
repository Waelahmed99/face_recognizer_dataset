class Colors:
    """
    Coloring text inside print function.
    How to use: pass Colors.HEADER (for example) before your colored string.
    To stop string coloring pass Colors.ENDC after it.
    """
    HEADER = '\033[95m'
    OK_BLUE = '\033[94m'
    OK_CYAN = '\033[96m'
    OK_GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

