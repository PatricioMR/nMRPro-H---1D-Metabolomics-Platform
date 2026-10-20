# NMRPro-H - 1D 1H NMR metabolomics
# Copyright (C) 2025-2026 Miguel Rocha Patricio
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""
NMRPro-H
Main entry point for the application
"""

import datetime as _dt
import sys
import os
import traceback


class _NullWriter:
    """Last-resort stand-in so nothing downstream ever writes to None."""

    encoding = 'utf-8'

    def write(self, _text):
        return len(_text) if isinstance(_text, str) else 0

    def flush(self):
        pass

    def isatty(self):
        return False


def _open_log_stream():
    """A real, writable stream for a build that has no console.

    Returns None only if even the fallback location is unwritable.
    """
    try:
        from app_paths import user_data_dir
        base = user_data_dir()
    except Exception:
        base = os.path.expanduser('~')
    try:
        return open(os.path.join(base, 'startup.log'), 'a', buffering=1,
                    encoding='utf-8', errors='replace')
    except Exception:
        return None


def _make_console_safe():
    """Keep startup logging from killing the app.

    A frozen build does not get a UTF-8 console: stdout is cp1252 on Windows,
    or None entirely when built with console=False. Printing a tick character
    then raises UnicodeEncodeError, and because the handler below also printed
    one, the app died after importing successfully.

    None is the worse case, and it is silent: print() with sys.stdout None is a
    no-op, but traceback.print_exception() calls .write on it and raises
    AttributeError. Tk routes every callback exception through that function, so
    a failure inside root.after() destroys the interpreter's only means of
    reporting it and mainloop() returns as if the user had closed the window.
    Pointing the missing streams at a log file makes the real traceback
    survivable and readable.
    """
    log = None
    for stream_name in ('stdout', 'stderr'):
        stream = getattr(sys, stream_name, None)
        if stream is None:
            if log is None:
                log = _open_log_stream() or _NullWriter()
            setattr(sys, stream_name, log)
            if getattr(sys, '__%s__' % stream_name, None) is None:
                setattr(sys, '__%s__' % stream_name, log)
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
print("NMRPro-H - Starting  %s" % _dt.datetime.now().isoformat(timespec='seconds'))
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