"""Environment smoke check; parser commands will follow in later milestones."""

import platform
import sys


def main() -> None:
    print("Multimodal document parser: environment ready")
    print(f"Python: {platform.python_version()}")
    print(f"Interpreter: {sys.executable}")


if __name__ == "__main__":
    main()
