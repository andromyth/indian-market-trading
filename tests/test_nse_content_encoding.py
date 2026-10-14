import nse_data_layer as layer


def test_nse_headers_only_advertise_supported_compression():
    encodings = {
        value.strip().lower()
        for value in layer.HEADERS["Accept-Encoding"].split(",")
    }
    assert encodings == {"gzip", "deflate"}
    assert "br" not in encodings


def test_nse_transport_diagnostic_records_content_encoding(monkeypatch):
    class Response:
        status_code = 200
        headers = {
            "content-type": "text/csv",
            "content-encoding": "gzip",
        }
        content = b"csv data"
        text = "csv data"

        def raise_for_status(self):
            return None

    class Session:
        def get(self, *args, **kwargs):
            return Response()

    monkeypatch.setattr(layer, "_session", lambda: Session())
    monkeypatch.setattr(layer, "_sleep_rate_limit", lambda *args, **kwargs: None)

    response = layer._request("/test", retries=1, min_interval=0)

    assert response.status_code == 200
    assert layer.nse_last_diagnostic()["content_encoding"] == "gzip"
