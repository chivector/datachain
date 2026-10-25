import io
import tarfile
import threading
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import Mock, mock_open, patch

import pytest
from fsspec.callbacks import Callback

from datachain.cache import Cache
from datachain.client.fsspec import Client
from datachain.client.http import HTTPClient, HTTPSClient
from datachain.lib.file import File
from datachain.lib.tar import process_tar


class _FalseyCallback(Callback):
    def __bool__(self):
        return False


def test_protocol_detection_https():
    client_class = Client.get_implementation("https://example.com/file.txt")
    assert client_class == HTTPSClient

    client_class = Client.get_implementation("http://example.com/file.txt")
    assert client_class == HTTPClient


def test_split_url():
    domain, path = HTTPClient.split_url("https://example.com/path/to/file.txt")
    assert domain == "example.com"
    assert path == "path/to/file.txt"

    # URL with query and fragment
    domain, path = HTTPClient.split_url("https://example.com/api?key=123#anchor")
    assert domain == "example.com"
    assert path == "api?key=123#anchor"


def test_is_root_url():
    assert HTTPClient.is_root_url("https://example.com")
    assert HTTPClient.is_root_url("https://example.com/")
    assert not HTTPClient.is_root_url("https://example.com/path")
    assert not HTTPClient.is_root_url("https://example.com?query=1")
    assert not HTTPClient.is_root_url("https://example.com#fragment")


def test_from_name_with_https():
    cache = Mock(spec=Cache)
    client = HTTPSClient.from_name("https://example.com/path", cache, {})
    assert client.protocol == "https"
    assert client.name == "example.com/path"
    assert client.PREFIX == "https://"


def test_from_name_with_http():
    cache = Mock(spec=Cache)
    client = HTTPClient.from_name("http://example.com/path", cache, {})
    assert client.protocol == "http"
    assert client.name == "example.com/path"
    assert client.PREFIX == "http://"


def test_get_uri_http():
    cache = Mock(spec=Cache)
    client = HTTPClient("example.com:8080", {}, cache)

    assert client.get_uri("file.txt") == "http://example.com:8080/file.txt"


def test_upload_raises_not_implemented():
    cache = Mock(spec=Cache)
    client = HTTPSClient("example.com", {}, cache)

    with pytest.raises(NotImplementedError, match="HTTP/HTTPS client is read-only"):
        client.upload(b"data", "path/to/file.txt")


def test_create_fs():
    from fsspec.implementations.http import HTTPFileSystem

    fs = HTTPClient.create_fs()
    assert isinstance(fs, HTTPFileSystem)

    fs = HTTPClient.create_fs(version_aware=True)
    assert isinstance(fs, HTTPFileSystem)
    assert not hasattr(fs, "version_aware")

    fs = HTTPClient.create_fs(timeout=30, headers={"User-Agent": "test"})
    assert isinstance(fs, HTTPFileSystem)

    fs = HTTPClient.create_fs()
    assert isinstance(fs, HTTPFileSystem)


def test_info_to_file():
    cache = Mock(spec=Cache)
    client = HTTPSClient("example.com", {}, cache)

    info = {
        "size": 1024,
        "ETag": '"abc123"',
        "last_modified": "Wed, 12 Oct 2024 07:28:00 GMT",
    }
    file = client.info_to_file(info, "path/to/file.txt")
    assert file.path == "path/to/file.txt"
    assert file.size == 1024
    assert file.etag == "abc123"

    # Test with int timestamp
    info = {"last_modified": 1700000000}
    file = client.info_to_file(info, "file1.txt")
    expected = datetime.fromtimestamp(1700000000, UTC)
    assert file.last_modified == expected

    # Test with float timestamp
    info = {"last_modified": 1700000000.5}
    file = client.info_to_file(info, "file2.txt")
    expected = datetime.fromtimestamp(1700000000.5, UTC)
    assert file.last_modified == expected

    # Test with invalid date string (triggers ValueError)
    info = {"last_modified": "invalid"}
    file = client.info_to_file(info, "file3.txt")
    assert isinstance(file.last_modified, datetime)  # Falls back to current time


def test_open_object(monkeypatch):
    cache = Mock(spec=Cache)
    client = HTTPSClient("example.com", {}, cache)

    file = File(source="https://example.com", path="path/to/file.txt")

    # Test with cache hit
    cache.get_path.return_value = "/cache/path/file.txt"
    with patch("builtins.open", mock_open(read_data=b"cached content")) as mock_file:
        with client.open_object(file, use_cache=True) as stream:
            assert stream.read() == b"cached content"
        mock_file.assert_called_once_with("/cache/path/file.txt", mode="rb")

    # Test without cache (cache miss)
    cache.get_path.return_value = None
    monkeypatch.setattr(client.fs, "open", Mock())
    client.open_object(file, use_cache=True)
    client.fs.open.assert_called_once()

    # Test with use_cache=False (bypass cache)
    cache.get_path.return_value = "/cache/path/file.txt"
    monkeypatch.setattr(client.fs, "open", Mock())
    client.open_object(file, use_cache=False)
    client.fs.open.assert_called_once()  # Should fetch from remote, not cache


@pytest.mark.parametrize("client_class", [HTTPClient, HTTPSClient])
@pytest.mark.parametrize("data", [b"", b"remote content"])
@pytest.mark.parametrize("callback_kwargs", [{}, {"cb": None}])
def test_open_object_reads_without_callback(
    monkeypatch, client_class, data, callback_kwargs
):
    cache = Mock(spec=Cache)
    cache.get_path.return_value = None
    client = client_class("example.com", {}, cache)
    monkeypatch.setattr(client.fs, "open", Mock(return_value=io.BytesIO(data)))
    file = File(source=client.uri, path="file.txt")

    with client.open_object(file, **callback_kwargs) as stream:
        assert stream.read(3) == data[:3]
        assert stream.read() == data[3:]
        assert stream.read() == b""


@pytest.mark.parametrize("client_class", [HTTPClient, HTTPSClient])
@pytest.mark.parametrize("callback_class", [Callback, _FalseyCallback])
def test_open_object_updates_callback(monkeypatch, client_class, callback_class):
    cache = Mock(spec=Cache)
    cache.get_path.return_value = None
    client = client_class("example.com", {}, cache)
    data = b"remote content"
    monkeypatch.setattr(client.fs, "open", Mock(return_value=io.BytesIO(data)))
    file = File(source=client.uri, path="file.txt")
    callback = callback_class()

    with client.open_object(file, cb=callback) as stream:
        assert stream.read(3) == data[:3]
        assert callback.value == 3
        assert stream.read() == data[3:]
        assert callback.value == len(data)
        assert stream.read() == b""
        assert callback.value == len(data)


def test_http_tar_member_read(catalog):
    data = b"remote tar member"
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        info = tarfile.TarInfo("member.txt")
        info.size = len(data)
        archive.addfile(info, io.BytesIO(data))
    raw = buffer.getvalue()

    class Handler(BaseHTTPRequestHandler):
        def do_HEAD(self):
            self.send_response(200)
            self.send_header("Content-Length", str(len(raw)))
            self.send_header("Accept-Ranges", "bytes")
            self.end_headers()

        def do_GET(self):
            start, end = 0, len(raw) - 1
            requested = self.headers.get("Range")
            if requested:
                first, last = requested.removeprefix("bytes=").split("-")
                start, end = int(first), min(int(last), end)
                self.send_response(206)
                self.send_header("Content-Range", f"bytes {start}-{end}/{len(raw)}")
            else:
                self.send_response(200)
            body = raw[start : end + 1]
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            source = f"http://127.0.0.1:{server.server_port}"
            parent = File(source=source, path="archive.tar")
            parent._set_stream(catalog, caching_enabled=False)
            member = next(process_tar(parent))
            member._set_stream(catalog, caching_enabled=False)

            assert member.read() == data
        finally:
            server.shutdown()
            thread.join()


def test_url():
    cache = Mock(spec=Cache)
    client = HTTPSClient("example.com", {}, cache)
    assert client.url("file.txt") == "https://example.com/file.txt"


def test_get_file_info(monkeypatch):
    cache = Mock(spec=Cache)
    client = HTTPSClient("example.com", {}, cache)
    monkeypatch.setattr(client.fs, "info", Mock(return_value={"size": 2048}))

    file = client.get_file_info("file.txt")
    assert file.path == "file.txt"
    assert file.size == 2048


@pytest.mark.asyncio
async def test_fetch_dir():
    cache = Mock(spec=Cache)
    client = HTTPSClient("example.com", {}, cache)

    with pytest.raises(NotImplementedError):
        await client._fetch_dir("prefix", None, None)


@pytest.mark.asyncio
async def test_get_file():
    cache = Mock(spec=Cache)
    client = HTTPSClient("example.com", {}, cache)

    async def mock_get_file(lpath, rpath, callback):
        return "data"

    client.fs._get_file = mock_get_file
    result = await client.get_file("/local", "/remote", None)
    assert result == "data"


def test_storage_uri():
    assert str(HTTPSClient.storage_uri("https://example.com")) == "https://example.com"
    assert str(HTTPSClient.storage_uri("example.com")) == "https://example.com"
