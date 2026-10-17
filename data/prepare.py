"""Turn text stories into token shards on disk.

Pipeline
    stories (text)  ->  GPT-2 BPE token ids  ->  fixed-size shards of uint16  ->  disk

Why shards?
    Tokenizing ~2 million TinyStories takes minutes, but training can be
    interrupted at any time (Colab sessions drop). Tokenizing once and saving the
    result means a restarted session just re-reads the files. Shards live in a
    configurable directory, e.g. a folder on Google Drive.

Why uint16?
    GPT-2's vocabulary has 50,257 ids, which fits in 16 bits (max 65,535). Half the
    disk and RAM of int32, and numpy can memory-map the files so only the pieces the
    loader touches are read.

Story boundaries
    Every story is followed by <|endoftext|> (id 50256). That is how the model
    learns where one story stops and the next begins, and it is the token the
    model emits when it "finishes".

Layers (each testable on its own, none needs the network except `load_tinystories`):
    story_tokens   text -> tokens
    write_shards   tokens -> shard files + a completion marker
    prepare_split  orchestration, skipped if the split is already prepared
"""
import argparse
import json
from pathlib import Path
from typing import Callable, Iterable, Iterator

import numpy as np
import tiktoken

# Name of the Hugging Face dataset (verified ID) and its text column.
DATASET_ID = "roneneldan/TinyStories"

# Tokens per shard. 100M uint16 tokens = 200 MB per file; the full train split
# (~470M tokens) becomes about 5 files.
DEFAULT_SHARD_SIZE = 100_000_000


def load_tinystories(split: str) -> Iterator[str]:
    """Yield the text of every story in a TinyStories split ("train" or "validation")."""
    from datasets import load_dataset  # imported here so the rest works without `datasets`

    dataset = load_dataset(DATASET_ID, split=split)
    return (row["text"] for row in dataset)


def story_tokens(stories: Iterable[str], batch: int = 1024) -> Iterator[np.ndarray]:
    """Yield one uint16 array of GPT-2 BPE ids per batch of stories, <|endoftext|> after each.

    encode_ordinary treats the text as plain text: a story that happens to contain
    the literal string "<|endoftext|>" is not mistaken for a real boundary. The real
    boundary id is appended explicitly. Batches are encoded on several threads.
    """
    enc = tiktoken.get_encoding("gpt2")
    eot = enc.eot_token  # 50256

    def encode_batch(texts):
        encoded = enc.encode_ordinary_batch(texts, num_threads=8)
        # Append <|endoftext|> to each story, then flatten into one array.
        return np.fromiter((t for ids in encoded for t in (*ids, eot)), dtype=np.uint16)

    texts = []
    for story in stories:
        texts.append(story)
        if len(texts) == batch:
            yield encode_batch(texts)
            texts = []
    if texts:
        yield encode_batch(texts)


def _marker(out_dir: Path, split: str) -> Path:
    return Path(out_dir) / f"{split}.done"


def write_shards(
    token_stream: Iterable[np.ndarray],
    out_dir,
    split: str,
    shard_size: int = DEFAULT_SHARD_SIZE,
) -> list[Path]:
    """Write a stream of token arrays as shards of exactly `shard_size` tokens.

    Every shard is full except possibly the last. A chunk may straddle two shards:
    shards are just a way of cutting one long token stream into files.
    A `<split>.done` marker is written LAST, so a run that crashes half-way leaves no
    marker and is redone from scratch next time (stale shards are deleted first).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Start clean: remove the marker first, then any shards from an earlier attempt.
    _marker(out_dir, split).unlink(missing_ok=True)
    for stale in out_dir.glob(f"{split}_*.npy"):
        stale.unlink()

    paths: list[Path] = []
    total = 0
    buffer = np.empty(shard_size, dtype=np.uint16)
    filled = 0

    def flush(n: int) -> None:
        path = out_dir / f"{split}_{len(paths):05d}.npy"
        np.save(path, buffer[:n])
        paths.append(path)

    for chunk in token_stream:
        chunk = np.asarray(chunk)
        if chunk.size and chunk.max() > np.iinfo(np.uint16).max:
            raise ValueError("token id does not fit in uint16")
        chunk = chunk.astype(np.uint16)
        total += chunk.size
        pos = 0
        while pos < chunk.size:
            take = min(shard_size - filled, chunk.size - pos)
            buffer[filled:filled + take] = chunk[pos:pos + take]
            filled += take
            pos += take
            if filled == shard_size:
                flush(filled)
                filled = 0
    if filled:
        flush(filled)

    _marker(out_dir, split).write_text(
        json.dumps({"num_shards": len(paths), "num_tokens": total, "shard_size": shard_size})
    )
    return paths


def shard_paths(out_dir, split: str) -> list[Path]:
    """Paths of a finished split's shards, in order. Fails if the split was never completed."""
    marker = _marker(out_dir, split)
    if not marker.exists():
        raise FileNotFoundError(f"split {split!r} is not prepared in {out_dir} (no {marker.name})")
    n = json.loads(marker.read_text())["num_shards"]
    return [Path(out_dir) / f"{split}_{i:05d}.npy" for i in range(n)]


def read_shards(out_dir, split: str) -> np.ndarray:
    """Load a whole split into one array. Convenient for small splits and tests;
    the training loader memory-maps shards instead of loading everything."""
    paths = shard_paths(out_dir, split)
    return np.concatenate([np.load(p) for p in paths]) if paths else np.empty(0, dtype=np.uint16)


def prepare_split(
    split: str,
    out_dir,
    get_stories: Callable[[], Iterable[str]] | None = None,
    shard_size: int = DEFAULT_SHARD_SIZE,
) -> list[Path]:
    """Tokenize one split into shards, unless it has already been done.

    get_stories is a function (not a list) so that nothing is downloaded or read
    when the shards already exist. It defaults to downloading TinyStories.
    """
    if _marker(out_dir, split).exists():
        return shard_paths(out_dir, split)  # already done: skip all the work
    stories = (get_stories or (lambda: load_tinystories(split)))()
    return write_shards(story_tokens(stories), out_dir, split, shard_size)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tokenize TinyStories into shards.")
    parser.add_argument("--out_dir", required=True, help="where shards are written (e.g. a Drive folder)")
    parser.add_argument("--splits", nargs="+", default=["train", "validation"])
    parser.add_argument("--shard_size", type=int, default=DEFAULT_SHARD_SIZE)
    args = parser.parse_args()
    for name in args.splits:
        paths = prepare_split(name, args.out_dir, shard_size=args.shard_size)
        print(f"{name}: {len(paths)} shard(s) in {args.out_dir}")
