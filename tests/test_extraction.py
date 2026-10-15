from pathlib import Path

from btp_usecase.services.extraction_service import TextExtractor, PdfExtractor, DocxExtractor


def test_txt_extractor_extracts_text():
    text = "Privacy review notes\nThis is a sample."
    result = TextExtractor().extract_text(text)
    assert result.text == text
    assert result.page_count == 1


def test_pdf_extractor_handles_empty_bytes():
    pdf = PdfExtractor()
    result = pdf.extract_file_content(b"not a real pdf")
    assert result.text == ""
    assert result.page_count == 0


def test_docx_extractor_supports_basic_document():
    doc = DocxExtractor()
    content = "A document for red teaming"
    text = "<w:p><w:r><w:t>{}</w:t></w:r></w:p>".format(content)
    result = doc.extract_text_from_xml(text)
    assert content in result
