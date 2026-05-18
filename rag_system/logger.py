import os
import json
import time


class Logger:
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.log_file   = os.path.join(output_dir, "experiment_log.txt")
        self.json_dir   = os.path.join(output_dir, "generated_questions")
        os.makedirs(self.json_dir, exist_ok=True)
        with open(self.log_file, 'w', encoding='utf-8') as f:
            f.write(f"Experiment Started at {time.ctime()}\n{'='*50}\n")

    def log(self, message: str):
        # Write all logs to the experiment log file only (no stdout).
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(message + "\n")

    def save_artifact(self, filename: str, data: dict):
        path = os.path.join(self.json_dir, filename)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
