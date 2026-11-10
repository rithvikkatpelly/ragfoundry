import threading
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import openpyxl
import pytest
from docx import Document as DocxDocument
from reportlab.pdfgen import canvas

from ragfoundry import IngestionError, load_sources
from ragfoundry.config import LoadingConfig, WebConfig
from ragfoundry.loaders import WebLoader


def test_text_markdown_and_csv(corpus):
    result = load_sources(corpus, LoadingConfig())
    by_type = {}
    for document in result.documents:
        by_type.setdefault(document.metadata["file_type"], []).append(document)
    assert set(by_type) == {"md", "txt", "csv"}
    assert len(by_type["csv"]) == 2
    assert "XQ-4471" in by_type["csv"][0].page_content


def test_pdf_pages(tmp_path):
    path = tmp_path / "report.pdf"
    pdf = canvas.Canvas(str(path))
    pdf.drawString(72, 720, "Quarterly revenue grew by twelve percent.")
    pdf.showPage()
    pdf.showPage()  # blank page, which must be skipped
    pdf.drawString(72, 720, "Operating costs fell.")
    pdf.save()
    documents = load_sources(path, LoadingConfig()).documents
    assert [d.page_content.strip() for d in documents] == [
        "Quarterly revenue grew by twelve percent.",
        "Operating costs fell.",
    ]
    assert documents[1].metadata["page"] == 2


def test_docx(tmp_path):
    path = tmp_path / "memo.docx"
    document = DocxDocument()
    document.add_paragraph("The launch moves to March.")
    document.save(path)
    documents = load_sources(path, LoadingConfig()).documents
    assert "launch moves to March" in documents[0].page_content


def test_excel_rows_keep_headers(tmp_path):
    path = tmp_path / "staff.xlsx"
    book = openpyxl.Workbook()
    sheet = book.active
    sheet.title = "People"
    sheet.append(["name", "role", "year"])
    sheet.append(["Ada", "engineer", 2021])
    sheet.append([None, None, None])
    sheet.append(["Lin", None, 2023])
    book.save(path)
    documents = load_sources(path, LoadingConfig()).documents
    assert [d.page_content for d in documents] == [
        "name: Ada\nrole: engineer\nyear: 2021",
        "name: Lin\nyear: 2023",
    ]
    assert documents[1].metadata == {
        "source": str(path),
        "sheet": "People",
        "row": 4,
        "file_type": "xlsx",
    }


def test_unsupported_and_missing_sources(tmp_path):
    (tmp_path / "slides.pptx").write_bytes(b"x")
    with pytest.raises(IngestionError, match="Unsupported file type"):
        load_sources(tmp_path / "slides.pptx", LoadingConfig())
    result = load_sources(
        [tmp_path / "slides.pptx", tmp_path / "nope.txt"], LoadingConfig(on_error="skip")
    )
    assert [Path(f.source).name for f in result.failures] == ["slides.pptx", "nope.txt"]


def test_scanned_pdf_reports_no_text(tmp_path):
    path = tmp_path / "scan.pdf"
    pdf = canvas.Canvas(str(path))
    pdf.showPage()
    pdf.save()
    result = load_sources(path, LoadingConfig(on_error="skip"))
    assert "No extractable text" in result.failures[0].error


@pytest.fixture
def site(tmp_path):
    (tmp_path / "index.html").write_text(
        "<html><head><title>Home</title></head><body><nav>Menu</nav><main><p>Welcome text.</p>"
        "<a href='/a.html'>A</a> <a href='/private/x.html'>X</a> <a href='/missing.html'>M</a>"
        "<a href='https://elsewhere.example/'>E</a></main><script>var x=1</script></body></html>"
    )
    (tmp_path / "a.html").write_text("<html><body><p>Page A body.</p></body></html>")
    (tmp_path / "private").mkdir()
    (tmp_path / "private" / "x.html").write_text("<html><body>secret</body></html>")
    (tmp_path / "robots.txt").write_text("User-agent: *\nDisallow: /private/\n")
    handler = partial(SimpleHTTPRequestHandler, directory=str(tmp_path))
    handler.log_message = lambda *args: None
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()


def test_web_single_page_strips_boilerplate(site):
    documents = WebLoader(f"{site}/index.html", WebConfig(delay_seconds=0)).load()
    assert len(documents) == 1
    assert documents[0].metadata["title"] == "Home"
    text = documents[0].page_content
    assert "Welcome text." in text
    assert "Menu" not in text
    assert "var x" not in text


def test_web_crawl_respects_robots_and_skips_dead_links(site):
    loader = WebLoader(f"{site}/index.html", WebConfig(max_depth=1, delay_seconds=0))
    sources = sorted(d.metadata["source"] for d in loader.load())
    assert sources == [f"{site}/a.html", f"{site}/index.html"]


def test_web_start_page_failure_raises(site):
    with pytest.raises(IngestionError, match="Could not fetch"):
        WebLoader(f"{site}/missing.html", WebConfig(delay_seconds=0)).load()
