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


def word_to_bytes_tuple(word: str) -> tuple[bytes]:
    return tuple(x for x in word.encode("utf-8"))


def _initialize_vocab(special_tokens: list[str]) -> tuple[dict[int, bytes], int]:
    vocab = {i : bytes([i]) for i in range(256)}
    special_token_bytes = [token.encode("utf-8") for token in special_tokens]
    next_id = 256

    # First initialize the dictionary, then add special tokens to the dictionary before BPE
    for token in special_token_bytes:
        vocab[next_id] = token
        next_id += 1

    return vocab, next_id


def _get_bytes_pair_freq(pre_token_freq: dict[tuple[bytes], int], pre_token_list: list[tuple[bytes]]) -> dict[tuple[bytes], int]:
    bytes_pair_freq = defaultdict(int)
    for pre_token_idx, pre_token in enumerate(pre_token_freq):
        freq = pre_token_freq[pre_token]
        for i in range(len(pre_token) - 1):
            byte_pair = (pre_token[i], pre_token[i + 1])
            bytes_pair_freq[byte_pair] += freq

    return bytes_pair_freq


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


# TODO: write train_bpe
def train_bpe(input_path: str, vocab_size: int, special_tokens: list[str]):
    return


