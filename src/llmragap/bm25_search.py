"""Search a list of texts with BM25, by free text or by keywords.

One-off search (builds the index each call):

    from llmragap.bm25_search import search

    search("How is the weather in London?", corpus)        # text: any sentence
    search(["windy", "London"], corpus)                    # keywords: a list
    search("windy London", corpus, mode="keyword")         # keywords: a string
    search("quite windy", corpus, mode="phrase")           # exact phrase only

Repeated searches over the same texts (build the index once):

    index = KeywordIndex(corpus)
    index.search("windy London", top_n=3)                  # -> [(position, score), ...]
"""

import re

from rank_bm25 import BM25Okapi

SEARCH_MODES = ("text", "keyword", "phrase")

# Common words that carry no meaning for search; dropped from text queries.
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "do", "for", "from", "how", "i",
    "in", "is", "it", "me", "of", "on", "or", "that", "the", "there", "this", "to",
    "was", "what", "when", "where", "which", "who", "why", "with", "you",
}


def tokenize(text: str) -> list[str]:
    """Lowercase and split into words, ignoring punctuation ("London?" -> "london")."""
    return re.findall(r"\w+", text.lower())


def query_terms(query: str | list[str], mode: str) -> tuple[list[str], str]:
    """Turn a query into search terms. A list of keywords always uses "keyword" mode."""
    if mode not in SEARCH_MODES:
        raise ValueError(f"mode must be one of {SEARCH_MODES}, got {mode!r}")
    if isinstance(query, list):
        return [t for word in query for t in tokenize(word)], "keyword"
    terms = tokenize(query)
    if mode == "text":
        terms = [t for t in terms if t not in STOPWORDS] or terms
    return terms, mode


class KeywordIndex:
    """BM25 index over a fixed list of texts; build once, search many times."""

    def __init__(self, corpus: list[str]):
        self.tokenized = [tokenize(doc) for doc in corpus]
        self.word_sets = [set(words) for words in self.tokenized]
        # BM25Okapi can't be built from an empty corpus.
        self.bm25 = BM25Okapi(self.tokenized) if corpus else None

    def __len__(self) -> int:
        return len(self.tokenized)

    def search(
        self, query: str | list[str], top_n: int = 3, mode: str = "text"
    ) -> list[tuple[int, float]]:
        """Return (position in corpus, score) for the best matches, best first.

        Modes:
            "text"    - free-text question; common words like "how", "is" are ignored.
            "keyword" - every word in the query counts as a keyword.
            "phrase"  - only texts containing the exact phrase (case-insensitive).
            A list of keywords is always searched in "keyword" mode.

        Texts that share no word with the query are left out, so fewer than `top_n`
        results (or none) can come back.
        """
        terms, mode = query_terms(query, mode)
        if not terms or self.bm25 is None:
            return []

        term_set = set(terms)
        phrase = " ".join(terms)
        scores = self.bm25.get_scores(terms)

        results = []
        for i, score in enumerate(scores):
            if not self.word_sets[i] & term_set:
                continue  # no word in common with the query
            if mode == "phrase" and phrase not in " ".join(self.tokenized[i]):
                continue  # phrase must appear as-is
            results.append((i, float(score)))

        results.sort(key=lambda r: r[1], reverse=True)
        return results[:top_n]


def search(
    query: str | list[str],
    corpus: list[str],
    top_n: int = 3,
    mode: str = "text",
) -> list[tuple[str, float]]:
    """Return the best-matching texts from `corpus` as (text, score), best first.

    Args:
        query: A sentence, a string of keywords, or a list of keywords.
        corpus: The texts to search.
        top_n: Maximum number of results.
        mode: "text", "keyword" or "phrase"; see KeywordIndex.search.
    """
    hits = KeywordIndex(corpus).search(query, top_n=top_n, mode=mode)
    return [(corpus[i], score) for i, score in hits]


if __name__ == "__main__":
    corpus = [
        "Hello there good man!",
        "It is quite windy in London",
        "How is the weather today?",
    ]
    print("text   :", search("Is it windy in London?", corpus, top_n=1))
    print("keyword:", search(["windy", "London"], corpus, top_n=1))
    print("keyword:", search("weather today", corpus, mode="keyword"))
    print("phrase :", search("quite windy", corpus, mode="phrase"))
    print("phrase :", search("windy quite", corpus, mode="phrase"))  # wrong order -> []
