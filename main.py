"""
NMRPro-H
Main entry point for the application
"""

import sys
import os
import traceback


def _make_console_safe():
    """Keep startup logging from killing the app.

    A frozen build does not get a UTF-8 console: stdout is cp1252 on Windows,
    or absent entirely when built with console=False. Printing a tick character
    then raises UnicodeEncodeError, and because the handler below also printed
    one, the app died after importing successfully — visible only in the
    packaged build, never when run from a development console.
    """
    for stream_name in ('stdout', 'stderr'):
        stream = getattr(sys, stream_name, None)
        if stream is None:
            continue
        try:
            stream.reconfigure(encoding='utf-8', errors='replace')
        except Exception:
            pass


def _has_console() -> bool:
    """True when there is a real stdin to wait on.

    A windowed build has none, so a "Press Enter to exit" prompt would raise or
    hang instead of showing the error it was meant to explain.
    """
    return bool(getattr(sys, 'stdin', None)) and sys.stdin.isatty()


def _pause_if_interactive():
    if _has_console():
        try:
            input("\nPress Enter to exit...")
        except Exception:
            pass


_make_console_safe()

print("=" * 50)
print("NMRPro-H - Starting")
print("=" * 50)

def main():
    """
    Main function to initialize and run the NMR viewer application.
    """
    try:
        # Add current directory to path for module imports
        current_dir = os.path.dirname(os.path.abspath(__file__))
        sys.path.append(current_dir)
        print(f"Current directory: {current_dir}")

        # Import the application class
        print("Importing App from gui.app...")
        from gui.app import App
        print("✓ App imported successfully")

        # Create and run the application
        print("Creating App instance...")
        app = App()
        print("✓ App instance created")

        print("Running application...")
        app.run()

    except ImportError as e:
        print(f"✗ Import Error: {e}")
        print("\nMake sure you have installed all dependencies:")
        print("  pip install numpy matplotlib pandas")
        print("\nAnd that the folder structure is correct:")
        print("  NMRAnalysis_Gui/")
        print("  ├── main.py")
        print("  ├── gui/")
        print("  │   ├── __init__.py")
        print("  │   └── app.py")
        print("  ├── data/")
        print("  │   ├── __init__.py")
        print("  │   └── loader.py")
        print("  └── plotting/")
        print("      ├── __init__.py")
        print("      └── spectrum.py")
        traceback.print_exc()
        _pause_if_interactive()
        sys.exit(1)

    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        traceback.print_exc()
        _pause_if_interactive()
        sys.exit(1)

if __name__ == "__main__":
    main()