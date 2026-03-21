import click

from factcheck_cli.explain import explain
from factcheck_cli.rag.add import add
from factcheck_cli.rag.check import check


@click.group()
def main():
    """factcheck-cli: ターミナルコマンド解説と論文ファクトチェックのCLIツール"""


main.add_command(explain)
main.add_command(add)
main.add_command(check)
