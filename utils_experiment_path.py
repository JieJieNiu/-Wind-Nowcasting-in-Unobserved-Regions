
import os
from datetime import datetime

def get_experiment_save_path(base_path, tag="default"):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"{tag}_{timestamp}"
    full_path = os.path.join(base_path, exp_name)
    os.makedirs(full_path, exist_ok=True)
    return full_path
