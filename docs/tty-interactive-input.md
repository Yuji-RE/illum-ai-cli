# bash `bind -x` 経由でのインタラクティブ入力問題

## 背景

`factcheck explain` コマンドは初回解説の後、対話形式でフォローアップ質問を受け付ける。
このコマンドは bash の `bind -x` キーバインドから呼び出される：

```bash
# ~/.bashrc
_factcheck_explain_readline() {
    factcheck explain "$READLINE_LINE"
}
bind -x '"\C-x\C-x": _factcheck_explain_readline'
```

## 問題

`bind -x` 経由でコマンドを起動すると、Python の `input()` が正常に動作しない。

### 根本原因

bash の readline は動作中、ターミナルを **raw モード**（`ECHO` OFF、`ICANON` OFF）に設定する。
`bind -x` でサブプロセスを起動するとき、bash はターミナルの設定を完全には復元しない。
そのため、サブプロセス内では：

- **`ECHO` が OFF** → 打鍵した文字がターミナルに表示されない
- **`ICANON` が OFF** → 行バッファリングが無効、Enter が効かない

### 試行と失敗の記録

| アプローチ | 結果 | 失敗理由 |
|---|---|---|
| `input()` (通常の stdin) | 入力不可 | stdin がターミナルではない |
| `sys.stdin = open("/dev/tty")` | 英語はEnterできるが文字が見えない、日本語は不可 | ECHO がOFFのまま |
| 上記 + `termios` で ECHO \| ICANON をセット | Enterも効かなくなった | ICANON を raw モードの attrs に追加すると他の設定と競合 |
| 上記 + `subprocess.run(["stty", "echo", "icanon"], stdin=tty_file)` | 変わらず動かない | `subprocess.run` の `stdin=` 経由では stty がターミナルを正しく認識しない |
| `os.system("stty echo icanon < /dev/tty")` | **解決** | シェルの `/dev/tty` リダイレクトで正しくターミナルに適用される |

## 解決策

```python
import os

# 1. 現在のターミナル設定を保存
saved_stty = os.popen("stty -g < /dev/tty 2>/dev/null").read().strip()

# 2. echo・行バッファリング・CR→NL変換を有効化
os.system("stty echo icanon icrnl < /dev/tty 2>/dev/null")

# 3. /dev/tty を stdin として開き input() を使う
tty_file = open("/dev/tty")
sys.stdin = tty_file

# --- input() による対話ループ ---

# 4. ターミナル設定を復元
tty_file.close()
os.system(f"stty {saved_stty} < /dev/tty 2>/dev/null")
sys.stdin = orig_stdin
```

### ポイント

- `stty` コマンドは **シェルの `< /dev/tty` リダイレクト**で実行することで、正しくターミナルデバイスに適用される
- `subprocess.run(..., stdin=tty_file)` では stty がターミナルと認識しないため効果がない
- `termios` の直接操作は raw モードの attrs との競合リスクがあるため避ける
- `stty -g` で保存した文字列はそのまま `stty <saved>` に渡せるため、確実な復元が可能
