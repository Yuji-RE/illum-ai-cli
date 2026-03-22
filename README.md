# illum-ai-cli

![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Ollama](https://img.shields.io/badge/powered%20by-Ollama-black)
![Local](https://img.shields.io/badge/inference-local%20only-orange)

A local-first AI CLI for shell command explanation and academic writing support, powered by Ollama.

## Features

| Command | Description |
|---------|-------------|
| `illum explain <cmd>` | Explain a shell command in Japanese (keyboard shortcut available — [see below](#explain--shell-command-explanation)) |
| `illum add <pdf>` | Index a PDF into the local reference database |
| `illum suggest <md>` | Suggest missing facts from your references as cited bullet points |

## Requirements

- Python 3.11+
- [uv](https://docs.astral.sh/uv/)
- [Ollama](https://ollama.com/) running locally

## Setup

```bash
git clone https://github.com/yourname/illum-ai-cli
cd illum-ai-cli
uv sync

# Pull required models
ollama pull qwen3:8b
ollama pull nomic-embed-text
```

- Add to `~/.bashrc`:

```bash
alias illum="$HOME/projects/illum-ai-cli/.venv/bin/illum"
```

## Usage

### explain — shell command explanation

Type a command at the shell prompt and press **Ctrl+x Ctrl+x** — the readline binding reads whatever is on the current prompt line and passes it to `illum explain` without executing it.

Add to `~/.bashrc`:

```bash
_illum_explain_readline() {
    local cmd="$READLINE_LINE"
    [[ -z "$cmd" ]] && return
    echo ""
    illum explain "$cmd"
}
bind -x '"\C-x\C-x": _illum_explain_readline'
```

You can also call it directly: `illum explain "grep -rn pattern src/"`

**Neovim integration** (`nvim/illum.lua`): `<leader>k` explains the word under the cursor or the visual selection in a floating window.

### add / suggest — academic writing support

```bash
# Index reference PDFs
illum add paper.pdf

# Suggest additions to your draft from indexed references
illum suggest draft.md
illum suggest draft.md -o suggestions.md
```

## Models

| Role | Model |
|------|-------|
| Text generation | qwen3:8b (`think=False`) |
| Embeddings | nomic-embed-text |

All inference runs locally — no API keys required.

## License

MIT
