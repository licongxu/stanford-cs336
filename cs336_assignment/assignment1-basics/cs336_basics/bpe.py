import os
from dataclasses import dataclass
import regex as re
import multiprocessing
import time
import pickle
import json
from typing import BinaryIO
from collections import defaultdict, Counter
from tqdm import tqdm

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


class Timer():
    def __init__(self):
        self.state = defaultdict(float)
        self.start_time = time.time()
        self.current_time = None

    def start(self) -> None:
        self.current_time = time.time()

    def tick(self, func_name: str) -> None:
        assert self.current_time is not None, 'timer need to start'
        interval_time = time.time() - self.current_time
        self.state[func_name] = interval_time
        self.current_time = time.time()

    def print_stats(self) -> None:
        self.state['total_time'] = time.time() - self.start_time
        for func_name, interval_time in self.state.items():
            print(f'{func_name}: {interval_time}')


def word_to_bytes_tuple(word: str) -> tuple[bytes, ...]:
    return tuple(bytes([b]) for b in word.encode("utf-8"))


def _initialize_vocab(special_tokens: list[str]) -> tuple[dict[int, bytes], int]:
    vocab = {i : bytes([i]) for i in range(256)}
    special_token_bytes = [token.encode("utf-8") for token in special_tokens]
    next_id = 256

    # First initialize the dictionary, then add special tokens to the dictionary before BPE
    for token in special_token_bytes:
        vocab[next_id] = token
        next_id += 1

    return vocab, next_id


def _count_pair_freq(pre_token_freq: dict[tuple[bytes, ...], int]) -> dict[tuple[bytes, bytes], int]:
    bytes_pair_freq: dict[tuple[bytes, bytes], int] = defaultdict(int)
    for pre_token, freq in pre_token_freq.items():
        for i in range(len(pre_token) - 1):
            byte_pair = (pre_token[i], pre_token[i + 1])
            bytes_pair_freq[byte_pair] += freq
    return bytes_pair_freq


def _get_bytes_pair_freq(
    pre_token_freq: dict[tuple[bytes, ...], int],
    pre_token_list: list[tuple[bytes, ...]],
) -> tuple[dict[tuple[bytes, bytes], int], dict[tuple[bytes, bytes], set[int]]]:
    bytes_pair_freq = _count_pair_freq(pre_token_freq)
    bytes_pair_map_to_pre_token_idx: dict[tuple[bytes, bytes], set[int]] = defaultdict(set)
    for pre_token_idx, pre_token in enumerate(pre_token_list):
        freq = pre_token_freq[pre_token]
        if freq == 0:
            continue
        for i in range(len(pre_token) - 1):
            byte_pair = (pre_token[i], pre_token[i + 1])
            bytes_pair_map_to_pre_token_idx[byte_pair].add(pre_token_idx)
    return bytes_pair_freq, bytes_pair_map_to_pre_token_idx


def _merge_pair_in_pre_token(
    pre_token: tuple[bytes, ...], pair: tuple[bytes, bytes]
) -> tuple[bytes, ...]:
    a, b = pair
    merged = a + b
    out: list[bytes] = []
    i = 0
    while i < len(pre_token):
        if i < len(pre_token) - 1 and pre_token[i] == a and pre_token[i + 1] == b:
            out.append(merged)
            i += 2
        else:
            out.append(pre_token[i])
            i += 1
    return tuple(out)


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

def _pretokenize_text(text: str, special_tokens: list[str]) -> dict[tuple[bytes, ...], int]:
    pre_token_freq: dict[tuple[bytes, ...], int] = {}
    if special_tokens:
        special_token_pat = "|".join(map(re.escape, special_tokens))
        sentences = re.split(special_token_pat, text)
    else:
        sentences = [text]
    for sentence in sentences:
        if not sentence:
            continue
        for word in (m.group(0) for m in re.finditer(PAT, sentence)):
            pre_token = word_to_bytes_tuple(word)
            pre_token_freq[pre_token] = pre_token_freq.get(pre_token, 0) + 1
    return pre_token_freq


def pretokenize_without_parallelizing(
    input_path: str | os.PathLike,
    special_tokens: list[str],
    num_chunk: int = 100,
) -> dict[tuple[bytes, ...], int]:
    pre_token_freq: dict[tuple[bytes, ...], int] = {}
    file_size = os.path.getsize(input_path)
    # Small corpora: one read is faster than chunk boundary scanning.
    if file_size <= 5 * 1024 * 1024:
        with open(input_path, encoding="utf-8") as f:
            text = f.read()
        return _pretokenize_text(text, special_tokens)

    split_token = special_tokens[0].encode("utf-8") if special_tokens else b"<|endoftext|>"
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_chunk, split_token)
        for start, end in zip(boundaries, boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            for pre_token, count in _pretokenize_text(chunk, special_tokens).items():
                pre_token_freq[pre_token] = pre_token_freq.get(pre_token, 0) + count
    return pre_token_freq


def pretokenize_with_parallelizing(
    input_path: str | os.PathLike, 
    special_tokens: list[str],
    num_chunk: int = 10,
) -> dict[tuple[bytes], int]:
    
    raise NotImplementedError


def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str],
    use_multiprocessing: bool = False,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    vocab, next_id = _initialize_vocab(special_tokens=special_tokens)
    merges: list[tuple[bytes, bytes]] = []

    pretoken_func = (
        pretokenize_with_parallelizing if use_multiprocessing else pretokenize_without_parallelizing
    )
    pre_token_freq = pretoken_func(input_path=input_path, special_tokens=special_tokens)

    num_merges = vocab_size - len(vocab)
    for _ in range(num_merges):
        bytes_pair_freq = _count_pair_freq(pre_token_freq)
        if not bytes_pair_freq:
            break
        max_freq = max(bytes_pair_freq.values())
        select_bytes_pair = max(k for k, v in bytes_pair_freq.items() if v == max_freq)
        merges.append(select_bytes_pair)
        merge_token = select_bytes_pair[0] + select_bytes_pair[1]
        vocab[next_id] = merge_token
        next_id += 1

        new_pre_token_freq: dict[tuple[bytes, ...], int] = defaultdict(int)
        for pre_token, freq in pre_token_freq.items():
            new_pre_token = _merge_pair_in_pre_token(pre_token, select_bytes_pair)
            new_pre_token_freq[new_pre_token] += freq
        pre_token_freq = dict(new_pre_token_freq)

    return vocab, merges


