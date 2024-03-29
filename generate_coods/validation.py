import json
import numpy as np
from pathlib import Path





def has_close_ships(annotations: list) -> bool:
    print(annotations[0].keys())
    print(annotations[0]['bbox'])
    ...