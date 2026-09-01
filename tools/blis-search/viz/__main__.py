"""`python -m viz` entry point, so the tool works from tools/blis-search/.

Running as a module puts the *parent* directory on sys.path, not viz/, so the
flat imports the rest of the package uses would fail without this insert.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cli import main                                              # noqa: E402

sys.exit(main())
