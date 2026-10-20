from typing import Any

class URDFLoadError(Exception):
    """Exception raised when a URDF file fails to load."""
    def __init__(self, message: str = "Failed to load URDF file.", value: Any = None) -> None:
        super().__init__(message)
        self.message = message
        self.value = value

    def __str__(self) -> str:
        return f"{self.message} (Value: {self.value})"