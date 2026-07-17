from typing import Callable

class PipelineLogger:
    def __init__(self, callback: Callable | None = None):
        self.callback = callback
        self.logs = []

    def log(self, level: str, message: str):
        self.logs.append({
            "level": level,
            "message": message
        })

        print(f"[{level}] {message}")

        if self.callback:
            self.callback(level, message)

    def info(self, message):
        self.log("INFO", message)

    def success(self, message):
        self.log("SUCCESS", message)

    def warning(self, message):
        self.log("WARNING", message)

    def error(self, message):
        self.log("ERROR", message)