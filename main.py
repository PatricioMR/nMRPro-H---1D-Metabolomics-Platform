"""
NMRPro-H
Main entry point for the application
"""

import sys
import os
import traceback

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
        input("\nPress Enter to exit...")
        sys.exit(1)

    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        traceback.print_exc()
        input("\nPress Enter to exit...")
        sys.exit(1)

if __name__ == "__main__":
    main()