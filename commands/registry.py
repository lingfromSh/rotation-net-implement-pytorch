from commands.command import ExitCommand
from commands.command_list import CommandList

main_command_list = CommandList(name="Main")
main_command_list.register_default_command(ExitCommand(name="Exit"))


def register_commands():
    from train import TrainCommand
    from validate import ValidateCommand
    main_command_list.register_command(TrainCommand(name="Training"))
    main_command_list.register_command(ValidateCommand(name="Validating"))


register_commands()
