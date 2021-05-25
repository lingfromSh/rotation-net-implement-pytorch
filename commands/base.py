from dataclasses import dataclass
from typing import Dict


@dataclass
class CommandInterface:
    name: str = "base"

    def command(self, context: Dict):
        raise NotImplementedError
