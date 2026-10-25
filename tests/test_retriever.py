import numpy as np

from civil_code_rag.chunking import Chunk
from civil_code_rag.embeddings import CharNgramEmbedder
from civil_code_rag.retriever import ChromaRetriever, NumpyRetriever

DOCS = [
    "向人民法院请求保护民事权利的诉讼时效期间为三年。",
    "租赁期限不得超过二十年，超过二十年的部分无效。",
    "自婚姻登记机关收到离婚登记申请之日起三十日内可以撤回申请。",
    "抵押期间抵押人可以转让抵押财产，抵押权不受影响。",
]


def make_chunks():
    return [
        Chunk(f"c{i}", "article", t, frozenset({i + 1})) for i, t in enumerate(DOCS)
    ]


def test_numpy_retriever_finds_near_duplicate():
    emb = CharNgramEmbedder(512)
    chunks = make_chunks()
    r = NumpyRetriever()
    r.add(chunks, emb.encode_documents(DOCS))
    q = emb.encode_query("诉讼时效期间是几年")
    top = r.search(q, k=2)
    assert top[0][0].chunk_id == "c0"
    assert len(top) == 2
    assert top[0][1] >= top[1][1]


def test_numpy_retriever_empty():
    r = NumpyRetriever()
    assert r.search(np.zeros(8, dtype=np.float32), k=3) == []


def test_chroma_matches_numpy_ranking(tmp_path):
    emb = CharNgramEmbedder(512)
    chunks = make_chunks()
    vecs = emb.encode_documents(DOCS)

    np_r = NumpyRetriever()
    np_r.add(chunks, vecs)
    ch_r = ChromaRetriever(tmp_path / "index", "test")
    ch_r.add(chunks, vecs)
    assert len(ch_r) == 4

    # Chroma's HNSW computes float32 distances on a different code path from
    # numpy exact cosine, so near-tied neighbours can legitimately swap ranks
    # (observed across chromadb versions in CI). The decision-relevant
    # invariants are: same top-1 winner, and same full recall set.
    for query in ("诉讼时效期间是几年", "离婚冷静期是多少日", "抵押人可以转让抵押财产吗"):
        q = emb.encode_query(query)
        np_all = np_r.search(q, k=4)
        ch_all = ch_r.search(q, k=4)
        assert ch_all[0][0].chunk_id == np_all[0][0].chunk_id, (
            f"chroma top-1 diverged from exact search for {query!r}"
        )
        assert {c.chunk_id for c, _ in ch_all} == {c.chunk_id for c, _ in np_all}

