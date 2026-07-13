import os
import sys

# Backend modules import each other bare (import catalog, from config import…),
# matching how Flask runs with --chdir backend — mirror that for tests.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))
