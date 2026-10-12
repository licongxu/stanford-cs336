from collections.abc import Iterable, Iterator

import regex as re

from cs336_basics.bpe import PAT


class Tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = sorted(
            special_tokens or [],
            key=len,
            reverse=True,
        )
        self.bytes_to_id = {b: i for i, b in vocab.items()}
        self.merge_ranks = {pair: i for i, pair in enumerate(merges)}

    def _encode_pretoken(self, piece: str) -> list[int]:
        parts = [bytes([b]) for b in piece.encode("utf-8")]
        while len(parts) >= 2:
            best_i = None
            best_rank = None
            for i in range(len(parts) - 1):
                rank = self.merge_ranks.get((parts[i], parts[i + 1]))
                if rank is None:
                    continue
                if best_rank is None or rank < best_rank:
                    best_rank = rank
                    best_i = i
            if best_i is None:
                break
            parts[best_i : best_i + 2] = [parts[best_i] + parts[best_i + 1]]
        return [self.bytes_to_id[b] for b in parts]

    def encode(self, text: str) -> list[int]:
        ids: list[int] = []
        i = 0
        while i < len(text):
            for special in self.special_tokens:
                if text.startswith(special, i):
                    ids.append(self.bytes_to_id[special.encode("utf-8")])
                    i += len(special)
                    break
            else:
                end = len(text)
                for special in self.special_tokens:
                    pos = text.find(special, i)
                    if pos != -1:
                        end = min(end, pos)
                for match in re.finditer(PAT, text[i:end]):
                    ids.extend(self._encode_pretoken(match.group(0)))
                i = end
        return ids

    def decode(self, ids: list[int]) -> str:
        return b"".join(self.vocab[i] for i in ids).decode("utf-8", errors="replace")

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        buffer = ""
        for chunk in iterable:
            buffer += chunk
            while buffer:
                hold = 0
                for special in self.special_tokens:
                    for n in range(1, len(special)):
                        if buffer.endswith(special[:n]):
                            hold = max(hold, n)
                if hold == len(buffer):
                    break
                safe = len(buffer) - hold
                segment, buffer = buffer[:safe], buffer[safe:]
                for token_id in self.encode(segment):
                    yield token_id
        for token_id in self.encode(buffer):
            yield token_id
