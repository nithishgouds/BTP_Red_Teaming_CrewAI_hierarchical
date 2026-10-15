from btp_usecase.storage.local import LocalDocumentStorage


def test_storage_save_and_retrieve(tmp_path):
    storage = LocalDocumentStorage(base_path=tmp_path / "documents")
    data = b"hello"
    document_id = "doc-123"

    storage.save(document_id, data)
    assert storage.get(document_id) == data

    storage.delete(document_id)
    assert storage.get(document_id) is None
