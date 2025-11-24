"""Tests for mcpdoc.main module."""

import pytest

from mcpdoc.main import (
    _get_fetch_description,
    _is_http_or_https,
    extract_domain,
)


def test_extract_domain() -> None:
    """Test extract_domain function."""
    # Test with https URL
    assert extract_domain("https://example.com/page") == "https://example.com/"

    # Test with http URL
    assert extract_domain("http://test.org/docs/index.html") == "http://test.org/"

    # Test with URL that has port
    assert extract_domain("https://localhost:8080/api") == "https://localhost:8080/"

    # Check trailing slash
    assert extract_domain("https://localhost:8080") == "https://localhost:8080/"

    # Test with URL that has subdomain
    assert extract_domain("https://docs.python.org/3/") == "https://docs.python.org/"


@pytest.mark.parametrize(
    "url,expected",
    [
        ("http://example.com", True),
        ("https://example.com", True),
        ("/path/to/file.txt", False),
        ("file:///path/to/file.txt", False),
        (
            "ftp://example.com",
            False,
        ),  # Not HTTP or HTTPS, even though it's not a local file
    ],
)
def test_is_http_or_https(url, expected):
    """Test _is_http_or_https function."""
    assert _is_http_or_https(url) is expected


@pytest.mark.parametrize(
    "has_local_sources,expected_substrings",
    [
        (True, ["local file path", "file://"]),
        (False, ["URL to fetch"]),
    ],
)
def test_get_fetch_description(has_local_sources, expected_substrings):
    """Test _get_fetch_description function."""
    description = _get_fetch_description(has_local_sources)

    # Common assertions for both cases
    assert "Fetch and parse documentation" in description
    assert "Returns:" in description

    # Specific assertions based on has_local_sources
    for substring in expected_substrings:
        if has_local_sources:
            assert substring in description
        else:
            # For the False case, we only check that "local file path"
            # and "file://" are NOT present
            if substring in ["local file path", "file://"]:
                assert substring not in description
