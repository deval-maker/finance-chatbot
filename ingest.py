"""Load html from files, clean up, split, ingest into Weaviate."""
import logging
import os
import re
from parser import langchain_docs_extractor
from langchain.document_loaders import GitLoader

import weaviate
from bs4 import BeautifulSoup, SoupStrainer
from langchain.document_loaders import RecursiveUrlLoader, SitemapLoader, DirectoryLoader, TextLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.indexes import SQLRecordManager, index
from langchain.schema.embeddings import Embeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.utils.html import (PREFIXES_TO_IGNORE_REGEX,
                                  SUFFIXES_TO_IGNORE_REGEX)
from langchain.vectorstores import Weaviate

from constants import WEAVIATE_DOCS_INDEX_NAME

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

WEAVIATE_URL = os.environ["WEAVIATE_URL"]
RECORD_MANAGER_DB_URL = os.environ["RECORD_MANAGER_DB_URL"]


def metadata_extractor(meta: dict, soup: BeautifulSoup) -> dict:
    title = soup.find("title")
    description = soup.find("meta", attrs={"name": "description"})
    html = soup.find("html")
    return {
        "source": meta["loc"],
        "title": title.get_text() if title else "",
        "description": description.get("content", "") if description else "",
        "language": html.get("lang", "") if html else "",
        **meta,
    }


def load_langchain_docs():
    logger.info("Fetching data for langchain docs")
    return SitemapLoader(
        "https://python.langchain.com/sitemap.xml",
        filter_urls=["https://python.langchain.com/"],
        parsing_function=langchain_docs_extractor,
        default_parser="lxml",
        bs_kwargs={
            "parse_only": SoupStrainer(
                name=("article", "title", "html", "lang", "content")
            ),
        },
        meta_function=metadata_extractor,
    ).load()

def load_ray_docs():
    logger.info("Fetching data for ray docs")
    return SitemapLoader(
        "https://docs.ray.io/en/latest/sitemap.xml",
        filter_urls=[],
        parsing_function=langchain_docs_extractor,
        default_parser="lxml",
        bs_kwargs={
            "parse_only": SoupStrainer(
                name=("article", "title", "html", "lang", "content")
            ),
        },
        meta_function=metadata_extractor,
    ).load()

def load_from_repo(repo_name, repo_path, branch="master"):
    logger.info(f"Fetching data from repo {repo_path}")
    return GitLoader(
        clone_url=repo_path,
        repo_path=f"./repos/{repo_name}",
        branch=branch,
    ).load()


def simple_extractor(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    return re.sub(r"\n\n+", "\n\n", soup.text).strip()


def load_api_docs():
    logger.info("Fetching data for langchain api docs")
    return RecursiveUrlLoader(
        url="https://api.python.langchain.com/en/latest/",
        max_depth=8,
        extractor=simple_extractor,
        prevent_outside=True,
        use_async=True,
        timeout=600,
        # Drop trailing / to avoid duplicate pages.
        link_regex=(
            f"href=[\"']{PREFIXES_TO_IGNORE_REGEX}((?:{SUFFIXES_TO_IGNORE_REGEX}.)*?)"
            r"(?:[\#'\"]|\/[\#'\"])"
        ),
        check_response_status=True,
        exclude_dirs=(
            "https://api.python.langchain.com/en/latest/_sources",
            "https://api.python.langchain.com/en/latest/_modules",
        ),
    ).load()


def get_embeddings_model() -> Embeddings:
    return OpenAIEmbeddings(chunk_size=200)


def ingest_docs(raw_docs):

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=200)
    docs_transformed = text_splitter.split_documents(raw_docs)

    # We try to return 'source' and 'title' metadata when querying vector store and
    # Weaviate will error at query time if one of the attributes is missing from a
    # retrieved document.
    for doc in docs_transformed:
        if "source" not in doc.metadata:
            doc.metadata["source"] = ""
        if "title" not in doc.metadata:
            doc.metadata["title"] = ""

    client = weaviate.Client(url = WEAVIATE_URL)
    embedding = get_embeddings_model()
    vectorstore = Weaviate(
        client=client,
        index_name=WEAVIATE_DOCS_INDEX_NAME,
        text_key="text",
        embedding=embedding,
        by_text=False,
        attributes=["source", "title"],
    )

    record_manager = SQLRecordManager(
        f"weaviate/{WEAVIATE_DOCS_INDEX_NAME}", db_url=RECORD_MANAGER_DB_URL
    )
    record_manager.create_schema()

    logger.info(f"docs_transformed: {len(docs_transformed)}")

    indexing_stats = index(
        docs_transformed,
        record_manager,
        vectorstore,
        cleanup=None,
        source_id_key="source",
    )

    logger.info("Indexing stats: ", indexing_stats)
    logger.info(
        "LangChain now has this many vectors: ",
        client.query.aggregate(WEAVIATE_DOCS_INDEX_NAME).with_meta_count().do(),
    )


if __name__ == "__main__":
    text_loader_kwargs={'autodetect_encoding': True}
    loaded_docs = [
        # load_langchain_docs(),
        # load_ray_docs(),
        # load_api_docs(),
        # DirectoryLoader("./repos/ray/doc/source/", loader_cls=TextLoader, loader_kwargs=text_loader_kwargs, silent_errors=True).load(),
        # DirectoryLoader("./repos/NeMo-Guardrails/docs/", loader_cls=TextLoader, loader_kwargs=text_loader_kwargs, silent_errors=True).load(),
        # DirectoryLoader("./repos/server/docs/", loader_cls=TextLoader, loader_kwargs=text_loader_kwargs, silent_errors=True).load(),
        DirectoryLoader("./repos/NeMo/docs/", loader_cls=TextLoader, loader_kwargs=text_loader_kwargs, silent_errors=True).load(),
    ]
    for raw_docs in loaded_docs:
        logger.info(f"Raw {len(raw_docs)} docs from documentation and apis")
        ingest_docs(raw_docs)
