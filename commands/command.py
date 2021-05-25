from __future__ import annotations

from typing import Dict

from commands.base import CommandInterface


class Command(CommandInterface):
    name: str = "blank"

    def command(self, context: Dict):
        raise NotImplementedError


class BackCommand(CommandInterface):
    name: str = "back"

    def command(self, context: Dict):
        context["prev"].command()


class ExitCommand(CommandInterface):
    name: str = "exit"

    def command(self, context: Dict):
        return
