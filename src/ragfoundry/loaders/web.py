"""Polite HTML scraper with optional same-site crawling."""

from __future__ import annotations

import logging
import re
import time
from collections import deque
from collections.abc import Iterator
from typing import Any
from urllib.parse import urldefrag, urljoin, urlparse
from urllib.robotparser import RobotFileParser

from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document

from .._utils import require
from ..config import WebConfig
from ..errors import IngestionError

logger = logging.getLogger(__name__)

_NON_CONTENT_TAGS = ["script", "style", "noscript", "nav", "header", "footer", "aside", "form"]
_TEXT_TYPES = ("text/html", "application/xhtml", "text/plain")


class WebLoader(BaseLoader):
    """Fetch a page and, when ``max_depth`` > 0, the pages it links to.

    Failures on the start URL raise; failures on discovered links are logged and skipped so one
    dead link does not abandon a crawl.
    """

    def __init__(self, url: str, config: WebConfig | None = None) -> None:
        self.url = url
        self.config = config or WebConfig()
        self._robots: dict[str, RobotFileParser | None] = {}

    def lazy_load(self) -> Iterator[Document]:
        httpx = require("httpx", "web")
        require("bs4", "web")
        config = self.config
        start_host = urlparse(self.url).netloc
        queue: deque[tuple[str, int]] = deque([(urldefrag(self.url)[0], 0)])
        seen = {queue[0][0]}
        fetched = 0

        with httpx.Client(
            follow_redirects=True,
            timeout=config.timeout_seconds,
            headers={"User-Agent": config.user_agent},
        ) as client:
            while queue and fetched < config.max_pages:
                url, depth = queue.popleft()
                if not self._allowed(client, url):
                    if depth == 0:
                        raise IngestionError(f"robots.txt disallows fetching {url}")
                    logger.info("robots.txt disallows %s", url)
                    continue
                if fetched:
                    time.sleep(config.delay_seconds)
                try:
                    html, content_type = self._fetch(client, url)
                except Exception as exc:
                    if depth == 0:
                        raise IngestionError(f"Could not fetch {url}: {exc}") from exc
                    logger.warning("Skipping %s: %s", url, exc)
                    continue
                fetched += 1
                document, links = _parse(url, html, content_type)
                if document.page_content.strip():
                    yield document
                if depth < config.max_depth:
                    for link in links:
                        if link not in seen and self._in_scope(link, start_host):
                            seen.add(link)
                            queue.append((link, depth + 1))

    def _in_scope(self, link: str, start_host: str) -> bool:
        return not self.config.same_domain or urlparse(link).netloc == start_host

    def _fetch(self, client: Any, url: str) -> tuple[str, str]:
        with client.stream("GET", url) as response:
            response.raise_for_status()
            content_type = response.headers.get("content-type", "").lower()
            if content_type and not content_type.startswith(_TEXT_TYPES):
                raise IngestionError(f"Unsupported content type '{content_type}'")
            body = bytearray()
            for piece in response.iter_bytes():
                body.extend(piece)
                if len(body) > self.config.max_bytes:
                    raise IngestionError(f"Response exceeds {self.config.max_bytes} bytes")
            charset = response.encoding or "utf-8"
        return bytes(body).decode(charset, errors="replace"), content_type

    def _allowed(self, client: Any, url: str) -> bool:
        if not self.config.respect_robots:
            return True
        parts = urlparse(url)
        origin = f"{parts.scheme}://{parts.netloc}"
        if origin not in self._robots:
            self._robots[origin] = _read_robots(client, origin)
        parser = self._robots[origin]
        return parser is None or parser.can_fetch(self.config.user_agent, url)


def _read_robots(client: Any, origin: str) -> RobotFileParser | None:
    try:
        response = client.get(f"{origin}/robots.txt")
    except Exception:
        return None
    if response.status_code != 200:
        return None
    parser = RobotFileParser()
    parser.parse(response.text.splitlines())
    return parser


def _parse(url: str, body: str, content_type: str) -> tuple[Document, list[str]]:
    if content_type.startswith("text/plain"):
        return Document(page_content=body.strip(), metadata={"source": url}), []

    from bs4 import BeautifulSoup

    soup = BeautifulSoup(body, "html.parser")
    links = []
    for anchor in soup.find_all("a", href=True):
        target = urldefrag(urljoin(url, str(anchor["href"])))[0]
        if urlparse(target).scheme in ("http", "https"):
            links.append(target)
    title = soup.title.get_text(strip=True) if soup.title else ""
    for tag in soup(_NON_CONTENT_TAGS):
        tag.decompose()
    root = soup.find("main") or soup.find("article") or soup.body or soup
    text = re.sub(r"\n\s*\n+", "\n\n", root.get_text("\n", strip=True))
    return Document(page_content=text, metadata={"source": url, "title": title}), links
