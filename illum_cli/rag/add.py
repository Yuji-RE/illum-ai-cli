import pathlib
import re

import click
import ollama
import pypdf

from illum_cli.rag.db import get_collection

EMBED_MODEL = "nomic-embed-text"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100


def chunk_text(text: str) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start : start + CHUNK_SIZE])
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return [c for c in chunks if c.strip()]


def _printed_page_num(text: str, fallback: int) -> int:
    """ページ先頭テキストから印刷ページ番号を推定する。
    先頭200文字に現れる最初の1〜4桁の独立した整数を使う。
    見つからなければ物理ページ番号（fallback）を返す。
    """
    for m in re.finditer(r"(?<!\d)(\d{1,4})(?!\d)", text[:200]):
        n = int(m.group(1))
        if 1 <= n <= 9999:
            return n
    return fallback


@click.command()
@click.argument("pdf_path", type=click.Path(exists=True, path_type=pathlib.Path))
def add(pdf_path: pathlib.Path):
    """PDFファイルをファクトチェックDBに登録します。

    使用例:\n
      factcheck add paper.pdf
    """
    click.echo(f"読み込み中: {pdf_path}")
    reader = pypdf.PdfReader(str(pdf_path))

    # ページごとにテキストを抽出してチャンク化（印刷ページ番号を保持）
    page_chunks: list[tuple[str, int]] = []  # (chunk_text, printed_page_number)
    for phys_num, page in enumerate(reader.pages, start=1):
        extracted = page.extract_text()
        if not extracted or not extracted.strip():
            continue
        printed = _printed_page_num(extracted, phys_num)
        for chunk in chunk_text(extracted):
            page_chunks.append((chunk, printed))

    if not page_chunks:
        raise click.ClickException("PDFからテキストを抽出できませんでした。")

    click.echo(f"{len(page_chunks)} チャンクに分割しました。埋め込み中...")

    collection = get_collection()
    source = str(pdf_path.resolve())

    with click.progressbar(enumerate(page_chunks), length=len(page_chunks), label="登録中") as bar:
        for i, (chunk, page_num) in bar:
            resp = ollama.embed(model=EMBED_MODEL, input=chunk)
            collection.add(
                documents=[chunk],
                embeddings=[resp.embeddings[0]],
                ids=[f"{source}::{i}"],
                metadatas=[{"source": source, "page": page_num, "chunk": i}],
            )

    click.echo(f"登録完了: {len(page_chunks)} チャンク ({source})")
