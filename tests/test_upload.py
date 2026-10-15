from io import BytesIO

import pytest
from fastapi import UploadFile

from btp_usecase.services.document_service import DocumentService
from btp_usecase.storage.local import LocalDocumentStorage


@pytest.fixture
def service(tmp_path):
    storage = LocalDocumentStorage(base_path=tmp_path / "documents")
    return DocumentService(storage=storage)


def test_upload_valid_txt_file(service):
    file = UploadFile(filename="sample.txt", file=BytesIO(b"hello world"))

    result = service.upload_document(file)

    assert result.document_id
    assert result.metadata.original_filename == "sample.txt"
    assert result.metadata.file_type == "txt"
    assert result.content == "hello world"


def test_reject_unsupported_extension(service):
    file = UploadFile(filename="sample.csv", file=BytesIO(b"hello"))

    with pytest.raises(ValueError, match="Unsupported file type"):
        service.upload_document(file)


def test_reject_oversized_file(service):
    service.max_file_size = 10
    file = UploadFile(filename="sample.txt", file=BytesIO(b"x" * 100))

    with pytest.raises(ValueError, match="too large"):
        service.upload_document(file)
