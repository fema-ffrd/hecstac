from unittest.mock import Mock

import obstore
import pytest

from hecstac.common.base_io import ModelFileReader, ModelFileReaderError


@pytest.mark.parametrize("environment_credentials", [False, True])
def test_s3_store_uses_default_credential_discovery(monkeypatch, environment_credentials):
    for name in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN"):
        if environment_credentials:
            monkeypatch.setenv(name, "test-credential")
        else:
            monkeypatch.delenv(name, raising=False)

    store = Mock()
    create_store = Mock(return_value=store)
    monkeypatch.setattr(obstore.store, "S3Store", create_store)
    reader = ModelFileReader.__new__(ModelFileReader)

    assert reader._create_s3_store("test-bucket") is store
    create_store.assert_called_once_with(bucket="test-bucket")


def test_s3_read_uses_supplied_store(monkeypatch):
    store = obstore.store.MemoryStore()
    obstore.put(store, "model.hms", b"Project: Test\r\nEnd:\r\n")
    create_store = Mock(side_effect=AssertionError("Must use supplied store"))
    monkeypatch.setattr(ModelFileReader, "_create_s3_store", create_store)

    reader = ModelFileReader("s3://test-bucket/model.hms", store=store)

    assert reader.content == "Project: Test\nEnd:\n"
    assert reader.store is store
    create_store.assert_not_called()


def test_s3_read_preserves_error_cause(monkeypatch):
    error = PermissionError("403 Forbidden")
    monkeypatch.setattr(obstore, "open_reader", Mock(side_effect=error))

    with pytest.raises(ModelFileReaderError, match="403 Forbidden") as exc_info:
        ModelFileReader("s3://test-bucket/model.hms", store=obstore.store.MemoryStore())

    assert exc_info.value.__cause__ is error