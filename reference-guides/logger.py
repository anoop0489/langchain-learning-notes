# =============================================================================
# CONSOLE LOGGER UTILITY: Colored + Emoji Output for Demo Scripts
# =============================================================================
# Provides readable colored terminal output for runnable scripts.
# Uses ANSI escape codes for color — works on modern terminals (Windows
# Terminal, VS Code, PowerShell 7+). Falls back gracefully on older terminals.
#
# USAGE:
#   from reference_guides.logger import log_info, log_success, log_error, log_warning, log_header
#
#   log_header("PIPELINE START")
#   log_info("Processing 50 documents...")
#   log_success("All batches stored!")
#   log_warning("3 batches failed, continuing...")
#   log_error("API key not found")
#
# WHY THIS EXISTS:
#   Eden's course uses a custom logger.py with ANSI colors. We adapted it
#   into a shared utility so any section's scripts can import it for
#   consistent, readable console output — especially useful for demos
#   and interview walkthroughs where you want the terminal to be clear.
#
# NOTE: Our section scripts also use inline emoji + print() which works
#   everywhere without imports. This utility is optional — use it when
#   you want colored output with consistent formatting.
# =============================================================================


class Colors:
    """ANSI escape codes for terminal colors."""
    PURPLE = "\033[95m"
    CYAN = "\033[96m"
    DARKCYAN = "\033[36m"
    BLUE = "\033[94m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    END = "\033[0m"


def log_info(message: str, color: str = Colors.CYAN):
    """Log an informational message (cyan by default)."""
    print(f"{color}ℹ️  {message}{Colors.END}")


def log_success(message: str):
    """Log a success message (green)."""
    print(f"{Colors.GREEN}✅ {message}{Colors.END}")


def log_error(message: str):
    """Log an error message (red)."""
    print(f"{Colors.RED}❌ {message}{Colors.END}")


def log_warning(message: str):
    """Log a warning message (yellow)."""
    print(f"{Colors.YELLOW}⚠️  {message}{Colors.END}")


def log_header(message: str):
    """Log a prominent header (purple, bold, with separators)."""
    print(f"\n{Colors.BOLD}{Colors.PURPLE}{'=' * 60}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.PURPLE}🚀 {message}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.PURPLE}{'=' * 60}{Colors.END}\n")


if __name__ == "__main__":
    # Quick demo — run this file directly to see the output
    log_header("LOGGER DEMO")
    log_info("This is an info message")
    log_info("Custom color info", Colors.BLUE)
    log_success("Operation completed successfully")
    log_warning("Something might be wrong")
    log_error("Something definitely went wrong")
    print()
    log_header("DONE")
