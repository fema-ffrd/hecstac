from unittest.mock import Mock

import pytest

from hecstac.ras.parser import GeometryHDFFile, PlanHDFFile, RASHDFFile


@pytest.mark.parametrize("reader_class", [RASHDFFile, PlanHDFFile, GeometryHDFFile])
@pytest.mark.parametrize("environment_credentials", [False, True])
def test_hdf_reader_uses_default_credential_discovery(monkeypatch, reader_class, environment_credentials):
    for name in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN"):
        if environment_credentials:
            monkeypatch.setenv(name, "test-credential")
        else:
            monkeypatch.delenv(name, raising=False)

    # Isolate the shared parser cache so every case exercises initialization.
    monkeypatch.setattr(reader_class, "_cache", {})
    hdf_object = Mock()
    open_uri = Mock(return_value=hdf_object)
    monkeypatch.setattr(reader_class._hdf_constructor, "open_uri", open_uri)
    path = "s3://test-bucket/model.hdf"

    reader = reader_class(path)

    assert reader.hdf_object is hdf_object
    open_uri.assert_called_once_with(
        path,
        fsspec_kwargs={"default_cache_type": "blockcache", "default_block_size": 10**5},
    )