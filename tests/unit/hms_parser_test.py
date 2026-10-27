from unittest.mock import Mock

import pytest

from hecstac.common.base_io import ModelFileReaderError
from hecstac.hms import parser


def test_project_read_preserves_s3_error(monkeypatch):
    error = ModelFileReaderError("403 Forbidden")
    monkeypatch.setattr(parser, "ModelFileReader", Mock(side_effect=error))

    with pytest.raises(ModelFileReaderError, match="403 Forbidden") as exc_info:
        parser.ProjectFile("s3://test-bucket/model.hms", recurse=False)

    assert exc_info.value is error


@pytest.mark.parametrize("encoding", ["utf-8", "cp1252"])
def test_project_read_local_file(tmp_path, encoding):
    path = tmp_path / "model.hms"
    content = "Project: Café\nEnd:\n"
    path.write_text(content, encoding=encoding)

    project = parser.ProjectFile(str(path), recurse=False)

    assert project.content == content
    assert project.name == "Café"