"""
Set Verbosity CLI Tool

Command-line utility for setting Protocol AI logging verbosity.

Usage:
    python set_verbosity.py debug    # Most verbose
    python set_verbosity.py info     # Standard
    python set_verbosity.py warning  # Only warnings and errors
    python set_verbosity.py error    # Only errors
    python set_verbosity.py critical # Only critical errors
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from protocol_ai_logging import set_verbosity, get_logger


def main():
    if len(sys.argv) != 2:
        print("Usage: python set_verbosity.py <level>")
        print("Levels: debug, info, warning, error, critical")
        sys.exit(1)

    level = sys.argv[1].lower()
    valid_levels = ['debug', 'info', 'warning', 'error', 'critical']

    if level not in valid_levels:
        print(f"Error: Invalid level '{level}'")
        print(f"Valid levels: {', '.join(valid_levels)}")
        sys.exit(1)

    set_verbosity(level)
    logger = get_logger()

    print(f"Verbosity set to: {level.upper()}")

    # Demonstrate different levels
    print("\nTesting log levels:")
    logger.debug("This is a DEBUG message")
    logger.info("This is an INFO message")
    logger.warning("This is a WARNING message")
    logger.error("This is an ERROR message")
    logger.critical("This is a CRITICAL message")

    print(f"\nYou should only see messages at {level.upper()} level and above.")


if __name__ == "__main__":
    main()
