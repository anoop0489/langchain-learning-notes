# =============================================================================
# EDEN'S ORIGINAL: Colored Console Logger (logger.py)
# =============================================================================
# Provides colored terminal output using ANSI escape codes.
# Used by ingestion.py for readable pipeline progress in the terminal.
#
# Our enhanced version lives at: reference-guides/logger.py
# (adds a __main__ demo block and expanded docblock)
#
# HOW ANSI COLORS WORK:
#   \033[ is the "escape" prefix, followed by a code number and 'm'
#   Example: \033[92m = bright green, \033[0m = reset to default
#   Works on modern terminals (Windows Terminal, VS Code, PowerShell 7+)
# =============================================================================


# ANSI escape codes — each string activates a color/style when printed
class Colors:
    PURPLE = "\033[95m"
    CYAN = "\033[96m"
    DARKCYAN = "\033[36m"
    BLUE = "\033[94m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    END = "\033[0m"       # Reset — always append after colored text


def log_info(message: str, color: str = Colors.CYAN):
    """Log info message with color"""
    print(f"{color}ℹ️  {message}{Colors.END}")


def log_success(message: str):
    """Log success message in green"""
    print(f"{Colors.GREEN}✅ {message}{Colors.END}")


def log_error(message: str):
    """Log error message in red"""
    print(f"{Colors.RED}❌ {message}{Colors.END}")


def log_warning(message: str):
    """Log warning message in yellow"""
    print(f"{Colors.YELLOW}⚠️  {message}{Colors.END}")


def log_header(message: str):
    """Log header message with emphasis"""
    print(f"\n{Colors.BOLD}{Colors.PURPLE}{'='*60}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.PURPLE}🚀 {message}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.PURPLE}{'='*60}{Colors.END}\n")
