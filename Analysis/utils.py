import os, sys
import pandas as pd
from pathlib import Path

def get_proj_root_path():
    return Path(__file__).parent.parent

def set_proj_root_path():
    return os.chdir(get_proj_root_path())
