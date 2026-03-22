import pathlib

import chromadb

DB_PATH = pathlib.Path.home() / ".local" / "share" / "factcheck-cli" / "db"


def get_collection() -> chromadb.Collection:
    client = chromadb.PersistentClient(path=str(DB_PATH))
    return client.get_or_create_collection("references")
