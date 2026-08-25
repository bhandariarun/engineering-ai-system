from dataclasses import dataclass
from collections import Counter
from math import log, sqrt
from pathlib import Path
import re


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    source: str
    text: str


class Retriever:
    def __init__(self, chunk_size: int = 900, overlap: int = 120) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.chunks: list[Chunk] = []
        self.document_vectors: list[dict[str, float]] = []
        self.idf: dict[str, float] = {}

    def ingest_directory(self, directory: str = "data") -> tuple[int, int]:
        files = sorted(Path(directory).glob("**/*"))
        documents = [path for path in files if path.is_file() and path.suffix.lower() in {".md", ".txt"}]
        self.chunks = [chunk for path in documents for chunk in self._chunk_file(path)]
        self._fit()
        return len(documents), len(self.chunks)

    def _chunk_file(self, path: Path) -> list[Chunk]:
        text = re.sub(r"\s+", " ", path.read_text(encoding="utf-8")).strip()
        result = []
        start = 0
        index = 0
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            result.append(Chunk(f"{path.name}:{index}", str(path), text[start:end]))
            if end == len(text):
                break
            start = max(end - self.overlap, start + 1)
            index += 1
        return result

    def _fit(self) -> None:
        tokenized = [self._tokens(chunk.text) for chunk in self.chunks]
        document_frequency = Counter(token for tokens in tokenized for token in set(tokens))
        count = len(tokenized)
        self.idf = {token: log((1 + count) / (1 + frequency)) + 1 for token, frequency in document_frequency.items()}
        self.document_vectors = [self._vector(tokens) for tokens in tokenized]

    @staticmethod
    def _tokens(text: str) -> list[str]:
        return re.findall(r"[a-z0-9]{2,}", text.lower())

    def _vector(self, tokens: list[str]) -> dict[str, float]:
        counts = Counter(tokens)
        total = max(len(tokens), 1)
        return {token: (frequency / total) * self.idf.get(token, 1) for token, frequency in counts.items()}

    @staticmethod
    def _cosine(left: dict[str, float], right: dict[str, float]) -> float:
        numerator = sum(value * right.get(token, 0) for token, value in left.items())
        denominator = sqrt(sum(value * value for value in left.values())) * sqrt(sum(value * value for value in right.values()))
        return numerator / denominator if denominator else 0.0

    def search(self, query: str, k: int = 4) -> list[tuple[Chunk, float]]:
        if not self.chunks:
            return []
        scores = [(self._cosine(self._vector(self._tokens(query)), vector), index) for index, vector in enumerate(self.document_vectors)]
        scores.sort(reverse=True)
        return [(self.chunks[index], score) for score, index in scores[:k] if score > 0]
