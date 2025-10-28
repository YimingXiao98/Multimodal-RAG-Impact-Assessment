from datetime import date
from pathlib import Path

from app.core.dataio.schemas import RAGQuery
from app.core.dataio.storage import DataLocator
from app.core.retrieval.context_packager import package_context
from app.core.retrieval.retriever import Retriever
from app.core.retrieval.selector import select_candidates


def test_retrieval_pipeline(tmp_path):
    locator = DataLocator(Path(__file__).resolve().parents[1] / "data")
    retriever = Retriever(locator)
    query = RAGQuery(zip="77002", start=date(2017, 8, 28), end=date(2017, 9, 3), k_tiles=4, n_text=5)
    result = retriever.retrieve(query)
    candidates = select_candidates(result, query.k_tiles, query.n_text)
    context = package_context(candidates)
    assert context["imagery_tiles"], "Imagery should not be empty"
    assert context["text_snippets"], "Text snippets expected"
