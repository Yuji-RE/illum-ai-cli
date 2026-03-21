import pathlib
import re
import sys
import unicodedata

import click
import ollama
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown

from factcheck_cli.rag.db import get_collection

EMBED_MODEL = "nomic-embed-text"
ORGANIZE_MODEL = "qwen3:8b"          # 提案モード: 指示追従重視
STRICT_MODEL = "qwen3:8b"            # 正誤判定: 指示追従重視
TOP_K_PER_SECTION = 3
MAX_CHUNKS_PER_SECTION = 3
MIN_BODY_LEN = 20
EMBED_MAX_CHARS = 4000

console = Console()

# セクション提案: 参考文献チャンクとドラフトのdiffから未記載情報を直接引用して箇条書きで出力
SECTION_ADVISE_PROMPT = """\
Compare <draft_section> with <reference_excerpts>. List at most 5 facts from <reference_excerpts> that add NEW information to the draft.

RULES:
1. Output ONLY bullet lines starting with "- ". No preamble. No headers. No numbered list.
2. Each bullet = direct quote or close paraphrase from <reference_excerpts> only.
3. Skip any fact whose MEANING is already covered in <draft_section> (even if worded differently).
4. Skip incomplete sentences (text that ends mid-sentence or gets cut off).
5. Max 5 bullets. Prefer the most specific, concrete facts.
6. End each bullet with the source label in the format `（SOURCE: label）` using the [SOURCE: ...] tag from the reference block.
7. If nothing genuinely new: output exactly "- 追加すべき情報なし".\
"""

# 正誤判定: ドキュメント全体を対象
EXTRACT_CLAIMS_PROMPT = """\
You are a claim extractor. Read the draft section and list every verifiable factual claim.
A claim is a statistic, an attributed quote, or a factual assertion that can be true or false.
Output one claim per line as a numbered list. Copy claims verbatim from the draft.
Do not add explanations. Output only the numbered list. Write in Japanese.\
"""

SINGLE_VERIFY_PROMPT = """\
You will be given a reference text and one claim. Decide if the reference supports, contradicts, or does not mention the claim.

Reply with EXACTLY one of these three formats. Nothing else. No explanation:
✅ 正しい — 「[short quote from reference]」
❌ 誤り — 「[short quote that contradicts]」
⚠️ 判断不能 — 参考文献チャンクに記載なし\
"""


# deepseek-r1が出力しやすい中国語簡体字 → 日本語漢字の対応表
_CHAR_FIXES: dict[str, str] = {
    "异": "異", "为": "ため", "实": "実", "这": "この",
    "时": "時", "对": "対", "间": "間", "动": "動",
    "从": "から", "们": "たち", "过": "過", "样": "様",
}


def fix_mixed_chars(text: str, author_corrections: dict[str, str] | None = None) -> str:
    """韓国語ハングルを除去し、既知の文字化けを修正する。"""
    # 韓国語ハングルを除去
    text = re.sub(r"[\uAC00-\uD7AF\u1100-\u11FF\uA960-\uA97F\uD7B0-\uD7FF]+", "", text)
    # 中国語簡体字 → 日本語漢字
    for cn, jp in _CHAR_FIXES.items():
        text = text.replace(cn, jp)
    # <sup>[著者年: ページ]</sup> → （著者年: ページ）
    text = re.sub(r"<sup>\[([^\]]+)\]</sup>", r"（\1）", text)
    # 残存する <sup>...</sup> タグを除去
    text = re.sub(r"<sup>(.*?)</sup>", r"\1", text)
    # 著者名の誤字を修正（ドラフト正確表記で上書き）
    if author_corrections:
        for wrong, correct in author_corrections.items():
            text = text.replace(wrong, correct)
    return text


def _drop_truncated_bullets(text: str) -> str:
    """文が途中で切れているバレット行を除去する。"""
    # 完全な文末とみなすパターン（ソースラベルを取り除いた後の内容で判定）
    _COMPLETE_END = re.compile(
        r"(。|．|[!！?？]|"
        r"である|であった|された|という|している|していた|とされる|とされた|"
        r"いる」|した」|ある」|れた」|きた」|する」)$"
    )
    lines = text.splitlines()
    kept: list[str] = []
    for line in lines:
        stripped = line.rstrip()
        if not stripped.startswith("- "):
            kept.append(line)
            continue
        content = stripped[2:].strip()
        # ソースラベル （SOURCE: ...） を末尾から除去して内容で判定
        content_without_source = re.sub(r"[（(]SOURCE:[^）)]+[）)]$", "", content).strip()
        # 「...」引用の場合は閉じ引用符の前の内容で判定
        inner = re.sub(r"^「|」$", "", content_without_source) if content_without_source.startswith("「") and content_without_source.endswith("」") else content_without_source
        # ソースラベルがあれば内容は完結しているとみなす
        has_source = bool(re.search(r"[（(]SOURCE:[^）)]+[）)]", content))
        if has_source or _COMPLETE_END.search(inner):
            kept.append(line)
        # それ以外は途中切れとみなして除外
    return "\n".join(kept)


def build_author_corrections(content: str) -> dict[str, str]:
    """ドラフトの著者名をもとに、1文字違いの誤字パターン → 正しい表記の辞書を返す。"""
    authors = extract_author_names(content)
    corrections: dict[str, str] = {}
    for name in authors:
        # 各文字を同じ読みの近隣漢字で置換したバリアントを登録
        # 既知の誤置換パターンのみ: 政↔正, 浩↔広, 明↔明 など
        _known_swaps = [("政", "正"), ("政", "正治"), ("浩", "広"), ("秀", "英")]
        for orig_char, wrong_char in _known_swaps:
            if orig_char in name:
                wrong_name = name.replace(orig_char, wrong_char)
                corrections[wrong_name] = name
    return corrections


def extract_author_names(content: str) -> list[str]:
    """ドラフトから著者名を抽出する（引用表記 著者（年）の著者部分）。"""
    matches = re.findall(
        r"([A-Za-z\u3040-\u30FF\u3400-\u9FFF\u4E00-\u9FFF]+)\s*[（(]\s*\d{4}",
        content,
    )
    return list(dict.fromkeys(m.strip() for m in matches if m.strip()))


def split_sections(content: str) -> list[tuple[str, str]]:
    """(header_line, body) のリストに分割。ヘッダーなし冒頭テキストは header='' で返す。"""
    parts = re.split(r"(^#{1,4} .+$)", content, flags=re.MULTILINE)
    sections: list[tuple[str, str]] = []

    if parts[0].strip():
        sections.append(("", parts[0].strip()))

    i = 1
    while i < len(parts):
        header = parts[i]
        body = parts[i + 1].strip() if i + 1 < len(parts) else ""
        sections.append((header, body))
        i += 2

    return sections


def fetch_chunks_for_text(text: str, collection) -> list[tuple[str, dict]]:
    """テキストに関連するチャンクを (document, metadata) のリストで返す（重複なし）。"""
    resp = ollama.embed(model=EMBED_MODEL, input=text[:EMBED_MAX_CHARS])
    results = collection.query(
        query_embeddings=[resp.embeddings[0]],
        n_results=min(TOP_K_PER_SECTION, collection.count()),
        include=["documents", "metadatas"],
    )
    seen: set[str] = set()
    chunks: list[tuple[str, dict]] = []
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        if doc not in seen and len(chunks) < MAX_CHUNKS_PER_SECTION:
            seen.add(doc)
            chunks.append((doc, meta or {}))
    return chunks


def _source_label(meta: dict) -> str:
    """メタデータからソースラベル文字列を生成する。例: 'paper.pdf p.5'"""
    source = pathlib.Path(meta.get("source", "")).name
    page = meta.get("page")
    return f"{source} p.{page}" if page else source


def call_model(messages: list, model: str = ORGANIZE_MODEL, think: bool = False) -> str:
    kwargs: dict = {"model": model, "messages": messages, "stream": True, "options": {"temperature": 0}}
    if not think:
        kwargs["think"] = False
    stream = ollama.chat(**kwargs)
    full_text = ""
    with Live(console=console, refresh_per_second=10, vertical_overflow="visible") as live:
        for chunk in stream:
            c = chunk.message.content
            if not c:
                continue
            full_text += c
            live.update(Markdown(full_text))
    return full_text


def organize_mode(content: str, collection) -> str:
    """セクションごとに参考文献との差分を箇条書きで返す。"""
    sections = split_sections(content)
    output_parts: list[str] = []

    for i, (header, body) in enumerate(sections):
        label = header.strip() if header else "（前文）"
        click.echo(f"  [{i + 1}/{len(sections)}] {label}")

        if len(body) < MIN_BODY_LEN:
            continue

        query_text = f"{header}\n{body}"
        chunk_pairs = fetch_chunks_for_text(query_text, collection)
        if not chunk_pairs:
            output_parts.append((f"{header}\n\n- 関連する参考文献チャンクなし").strip() if header else "- 関連する参考文献チャンクなし")
            continue

        # ソースラベル付きでexcerptsを構築
        excerpt_blocks = [
            f"[SOURCE: {_source_label(meta)}]\n{doc}"
            for doc, meta in chunk_pairs
        ]
        excerpts = "\n\n---\n\n".join(excerpt_blocks)

        console.print(f"\n[bold]{label}[/bold]")
        result = call_model([
            {"role": "system", "content": SECTION_ADVISE_PROMPT},
            {
                "role": "user",
                "content": (
                    f"<reference_excerpts>\n{excerpts}\n</reference_excerpts>\n\n"
                    f"<draft_section>\n{header}\n\n{body}\n</draft_section>"
                ),
            },
        ])
        bullets = fix_mixed_chars(result.strip())
        bullets = _drop_truncated_bullets(bullets)
        section_block = f"{header}\n\n{bullets}" if header else bullets
        output_parts.append(section_block)
        console.print()

    return "\n\n".join(output_parts)


def extract_verdict(text: str) -> str:
    """モデル出力から**判定**/**根拠**行を抽出する。長文エッセイが出ても正しく取れる。"""
    lines = [l.strip() for l in text.splitlines()]
    result: list[str] = []
    for line in lines:
        if line.startswith("**判定**") or line.startswith("**根拠**"):
            result.append(line)
    if result:
        return "\n".join(result)
    # フォールバック: 絵文字を含む行を探す
    for line in lines:
        if any(e in line for e in ("✅", "❌", "⚠️")):
            return line
    return "**判定**: ⚠️ 判断不能\n**根拠**: 参考文献チャンクに記載なし"


def call_model_silent(messages: list, model: str = STRICT_MODEL) -> str:
    """ストリーミングなしでモデルを呼び出す（クレーム抽出など内部処理用）。"""
    resp = ollama.chat(
        model=model,
        messages=messages,
        stream=False,
        think=False,
        options={"temperature": 0},
    )
    return resp.message.content or ""


def strict_mode(content: str, collection) -> str:
    """セクションごとに2段階（クレーム抽出→個別検証）でファクトチェックする。"""
    sections = split_sections(content)
    output_parts: list[str] = []

    for i, (header, body) in enumerate(sections):
        label = header.strip() if header else "（前文）"
        click.echo(f"  [{i + 1}/{len(sections)}] {label}")

        if len(body) < MIN_BODY_LEN:
            continue

        # Step 1: クレーム抽出（表示なし）
        claims_raw = call_model_silent([
            {"role": "system", "content": EXTRACT_CLAIMS_PROMPT},
            {"role": "user", "content": f"<draft_section>\n{header}\n\n{body}\n</draft_section>"},
        ])
        claims = [
            re.sub(r"^\d+[\.\)]\s*", "", line).strip()
            for line in claims_raw.splitlines()
            if re.match(r"^\d+[\.\)]", line.strip())
        ]
        if not claims:
            click.echo("    クレームなし")
            continue

        click.echo(f"    {len(claims)} クレームを検出、検証中...")

        if header:
            output_parts.append(header)

        for claim in claims:
            # クレームごとに最も関連するチャンクを取得
            resp = ollama.embed(model=EMBED_MODEL, input=claim[:EMBED_MAX_CHARS])
            results = collection.query(
                query_embeddings=[resp.embeddings[0]],
                n_results=min(2, collection.count()),
                include=["documents"],
            )
            claim_chunks = results["documents"][0]
            excerpts = "\n\n---\n\n".join(claim_chunks) if claim_chunks else "（関連チャンクなし）"

            raw = call_model_silent([
                {"role": "system", "content": SINGLE_VERIFY_PROMPT},
                {
                    "role": "user",
                    "content": f"参考文献:\n{excerpts}\n\nクレーム: {claim}",
                },
            ])
            verdict_line = fix_mixed_chars(raw.strip().splitlines()[0] if raw.strip() else "")
            if not any(e in verdict_line for e in ("✅", "❌", "⚠️")):
                verdict_line = "⚠️ 判断不能 — 参考文献チャンクに記載なし"

            output_parts.append(f"**クレーム**: {claim}\n{verdict_line}")
            click.echo(f"    {'✅' if '✅' in verdict_line else '❌' if '❌' in verdict_line else '⚠️'} {claim[:40]}...")

    return "\n\n".join(output_parts)


@click.command()
@click.argument("md_file", type=click.Path(exists=True), required=False)
@click.option("--strict", is_flag=True, help="正誤判定モード（✅/❌/⚠️ で各クレームを判定）")
@click.option("-o", "--output", type=click.Path(), default=None, help="出力先ファイル（省略時は refs/{入力名}_{mode}.md）")
def check(md_file, strict, output):
    """Markdownファイルのクレームを参考文献DBと照合してファクトチェックします。

    使用例:\n
      factcheck check note.md\n
      factcheck check --strict note.md\n
      factcheck check -o result.md note.md
    """
    if md_file:
        input_path = pathlib.Path(md_file)
        with open(input_path, encoding="utf-8") as f:
            content = f.read().strip()
    elif not sys.stdin.isatty():
        input_path = None
        content = sys.stdin.read().strip()
    else:
        raise click.UsageError(
            "Markdownファイルを引数またはパイプで渡してください。\n"
            "例: factcheck check note.md"
        )

    if not content:
        raise click.UsageError("入力が空です。")

    mode_suffix = "strict" if strict else "suggest"
    refs_dir = pathlib.Path.cwd() / "refs"
    refs_dir.mkdir(exist_ok=True)
    if output:
        output_path = pathlib.Path(output)
    elif input_path:
        output_path = refs_dir / f"{input_path.stem}_{mode_suffix}.md"
    else:
        output_path = refs_dir / f"factcheck_{mode_suffix}.md"

    mode_label = "正誤判定モード" if strict else "提案モード"

    try:
        collection = get_collection()
        if collection.count() == 0:
            raise click.ClickException(
                "DBに参考文献が登録されていません。\n"
                "先に `factcheck add <pdf>` で論文を登録してください。"
            )

        click.echo(f"[{mode_label}] 処理開始...")

        if strict:
            result = strict_mode(content, collection)
        else:
            result = organize_mode(content, collection)

        output_path.write_text(result, encoding="utf-8")
        click.echo(f"\n結果を保存しました: {output_path}")

    except ollama.ResponseError as e:
        if "model" in str(e).lower() and "not found" in str(e).lower():
            click.echo(
                f"エラー: モデル '{CHECK_MODEL}' が見つかりません。\n"
                f"以下のコマンドで取得してください:\n  ollama pull {CHECK_MODEL}",
                err=True,
            )
        else:
            click.echo(f"Ollama エラー: {e}", err=True)
        sys.exit(1)
    except click.ClickException:
        raise
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
