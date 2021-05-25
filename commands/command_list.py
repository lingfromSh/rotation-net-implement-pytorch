from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

from PyInquirer import prompt

from commands.base import CommandInterface


@dataclass
class CommandList(CommandInterface):
    commands: Dict = field(default_factory=dict)
    default_commands: Dict = field(default_factory=dict)

    def register_default_command(self, command: CommandInterface):
        self.default_commands[command.name] = command

    def register_command(self, command: CommandInterface):
        self.commands[command.name] = command

    def get_command_names(self):
        return list(self.commands.keys()) + list(self.default_commands.keys())

    def command(self, context=None):
        if context is None:
            context = {}

        menu = {
            "name": "choice",
            "message": "Command list",
            "type": "list",
            "choices": self.get_command_names()
        }
        if not self.get_command_names():
            print("Empty commands")
            return
        choice = prompt(menu).get("choice")
        command = self.commands.get(choice)

        if isinstance(command, CommandList):
            context.update(prev=self)
        if command:
            self.commands[choice].command(context=context)
