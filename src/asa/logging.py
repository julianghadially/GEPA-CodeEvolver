from typing import Protocol


class LoggerProtocol(Protocol):
    def log(self, message: str) -> None: ...


class StdOutLogger:
    def log(self, message: str) -> None:
        print(message)
