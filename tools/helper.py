import os
from pathlib import Path

class Helper:
    @staticmethod
    def find_root_path(marker='.git'):
        current_path = Path(os.getcwd())
        for path in [current_path] + list(current_path.parents):
            if (path / marker).exists():
                return path
        raise FileNotFoundError(f"Could not find the root directory containing {marker}")