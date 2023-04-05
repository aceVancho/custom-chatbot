import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

import requests

from llama_index.readers.base import BaseReader
from llama_index.readers.schema.base import Document

logger = logging.getLogger(__name__)

def _substack_reader(soup: Any) -> Tuple[str, Dict[str, Any]]:
    """Extract text from Substack blog post."""
    extra_info = {
        "Title of this Substack post": soup.select_one("h1.post-title").getText(),
        "Subtitle": soup.select_one("h3.subtitle").getText(),
        "Author": soup.select_one("span.byline-names").getText(),
    }
    text = soup.select_one("div.available-content").getText()
    return text, extra_info

DEFAULT_WEBSITE_EXTRACTOR: Dict[str, Callable[[Any], Tuple[str, Dict[str, Any]]]] = {
    "substack.com": _substack_reader,
}
class Custom_BeautifulSoupWebReader(BaseReader):
    """BeautifulSoup web page reader.

    Reads pages from the web.
    Requires the `bs4` and `urllib` packages.

    Args:
        file_extractor (Optional[Dict[str, Callable]]): A mapping of website
            hostname (e.g. google.com) to a function that specifies how to
            extract text from the BeautifulSoup obj. See DEFAULT_WEBSITE_EXTRACTOR.
    """

    def __init__(
        self,
        website_extractor: Optional[Dict[str, Callable]] = None,
    ) -> None:
        """Initialize with parameters."""
        try:
            from urllib.parse import urlparse  # noqa: F401

            import requests  # noqa: F401
            from bs4 import BeautifulSoup  # noqa: F401
        except ImportError:
            raise ImportError(
                "`bs4`, `requests`, and `urllib` must be installed to scrape websites."
                "Please run `pip install bs4 requests urllib`."
            )

        self.website_extractor = website_extractor or DEFAULT_WEBSITE_EXTRACTOR

    def load_data(
        self,
        urls: List[str],
        custom_hostname: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
    ) -> List[Document]:
        from urllib.parse import urlparse

        import requests
        from bs4 import BeautifulSoup
        from requests.auth import HTTPBasicAuth

        session = requests.Session()
        if username and password:
            session.auth = HTTPBasicAuth(username, password)

        documents = []
        for url in urls:
            try:
                page = session.get(url)
            except Exception:
                raise ValueError(f"One of the inputs is not a valid url: {url}")

            hostname = custom_hostname or urlparse(url).hostname or ""

            soup = BeautifulSoup(page.content, "html.parser")

            data = ""
            extra_info = {"URL": url}
            if hostname in self.website_extractor:
                data, metadata = self.website_extractor[hostname](soup)
                extra_info.update(metadata)
            else:
                data = soup.getText()

            documents.append(Document(data, extra_info=extra_info))

        return documents