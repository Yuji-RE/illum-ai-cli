import os
import sys

import click
import ollama
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown

MODEL = "qwen3:8b"
console = Console()

SYSTEM_PROMPT = (
    "あなたはターミナル操作の専門家です。"
    "ユーザーが入力したシェルコマンドや操作について、日本語で簡潔かつ正確に解説してください。"
    "コマンドの目的、主要なオプション、典型的な使用例を含めて説明してください。"
    "余計な前置きは不要です。すぐに解説を始めてください。"
)


def stream_response(messages: list) -> str:
    stream = ollama.chat(
        model=MODEL,
        messages=messages,
        stream=True,
        think=False,
    )
    full_text = ""
    with Live(console=console, refresh_per_second=10, vertical_overflow="visible") as live:
        for chunk in stream:
            content = chunk.message.content
            if not content:
                continue
            full_text += content
            live.update(Markdown(full_text))
    return full_text


def stream_explain(query: str) -> None:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": query},
    ]
    try:
        response = stream_response(messages)
        messages.append({"role": "assistant", "content": response})

        orig_stdin = sys.stdin
        tty_file = None
        saved_stty = ""
        try:
            saved_stty = os.popen("stty -g < /dev/tty 2>/dev/null").read().strip()
            os.system("stty echo icanon icrnl < /dev/tty 2>/dev/null")
            tty_file = open("/dev/tty")
            sys.stdin = tty_file
        except OSError:
            pass

        while True:
            try:
                sys.stdout.write("\n\033[43m\033[30m  質問>  \033[0m ")
                sys.stdout.flush()
                line = sys.stdin.readline()
                if not line:
                    break
                follow_up = line.strip()
            except KeyboardInterrupt:
                print()
                break
            if not follow_up:
                break
            messages.append({"role": "user", "content": follow_up})
            console.print()
            response = stream_response(messages)
            messages.append({"role": "assistant", "content": response})

        if tty_file:
            tty_file.close()
        if saved_stty:
            os.system(f"stty {saved_stty} < /dev/tty 2>/dev/null")
        sys.stdin = orig_stdin

    except ollama.ResponseError as e:
        if "model" in str(e).lower() and "not found" in str(e).lower():
            click.echo(
                f"エラー: モデル '{MODEL}' が見つかりません。\n"
                f"以下のコマンドで取得してください:\n  ollama pull {MODEL}",
                err=True,
            )
        else:
            click.echo(f"Ollama エラー: {e}", err=True)
        sys.exit(1)
    except Exception as e:
        if "connection" in str(e).lower() or "connect" in str(e).lower():
            click.echo(
                "エラー: Ollama に接続できません。\n"
                "Ollama が起動しているか確認してください:\n  ollama serve",
                err=True,
            )
        else:
            click.echo(f"エラー: {e}", err=True)
        sys.exit(1)


@click.command()
@click.argument("query", nargs=-1)
def explain(query):
    """シェルコマンドを日本語で解説します。

    使用例:\n
      factcheck explain awk\n
      factcheck explain "grep -rn pattern dir/"\n
      echo "ps aux" | factcheck explain
    """
    if query:
        q = " ".join(query)
    elif not sys.stdin.isatty():
        q = sys.stdin.read().strip()
        if not q:
            raise click.UsageError("入力が空です。")
    else:
        raise click.UsageError(
            "コマンドを引数またはパイプで渡してください。\n"
            "例: factcheck explain ls\n"
            "例: echo 'ps aux' | factcheck explain"
        )

    stream_explain(q)
