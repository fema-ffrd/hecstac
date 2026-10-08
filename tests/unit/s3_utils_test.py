from hecstac.common.s3_utils import make_uri_public


def test_make_uri_public_encodes_special_characters_in_key():
    uri = "s3://bucket/path/my file?draft#version%.txt"

    assert make_uri_public(uri) == ("https://bucket.s3.amazonaws.com/path/my%20file%3Fdraft%23version%25.txt")


def test_make_uri_public_does_not_double_encode_escaped_key():
    uri = "s3://bucket/path/my%20file.txt"

    assert make_uri_public(uri) == "https://bucket.s3.amazonaws.com/path/my%20file.txt"
