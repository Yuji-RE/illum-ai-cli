import click

from illum_cli.explain import explain
from illum_cli.rag.add import add
from illum_cli.rag.check import suggest_cmd


@click.group()
def main():
    """illum: AIによる研究・開発支援CLIツール"""


main.add_command(explain)
main.add_command(add)
main.add_command(suggest_cmd, name="suggest")
