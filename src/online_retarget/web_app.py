"""Standard-library web console for local motion retarget previews."""

from __future__ import annotations

from email.parser import BytesParser
from email.policy import default
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import argparse
import json
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from .web_pipeline import DEFAULT_G1_MJCF, WEB_RUN_ROOT, run_web_pipeline


STATIC_DIR = Path(__file__).resolve().parent / "web_static"


class RetargetWebHandler(BaseHTTPRequestHandler):
    server_version = "OnlineRetargetWeb/0.1"

    def do_GET(self) -> None:  # noqa: N802 - stdlib handler API.
        parsed = urlparse(self.path)
        if parsed.path in {"", "/"}:
            self._send_file(STATIC_DIR / "index.html", "text/html; charset=utf-8")
            return
        if parsed.path == "/app.js":
            self._send_file(STATIC_DIR / "app.js", "text/javascript; charset=utf-8")
            return
        if parsed.path == "/styles.css":
            self._send_file(STATIC_DIR / "styles.css", "text/css; charset=utf-8")
            return
        if parsed.path == "/favicon.ico":
            self.send_response(HTTPStatus.NO_CONTENT)
            self.end_headers()
            return
        if parsed.path == "/api/result":
            run_id = parse_qs(parsed.query).get("run_id", [""])[0]
            self._send_result(run_id)
            return
        if parsed.path == "/api/artifact":
            query = parse_qs(parsed.query)
            run_id = query.get("run_id", [""])[0]
            artifact_name = query.get("name", [""])[0]
            self._send_artifact(run_id, artifact_name)
            return
        self._send_json({"error": "not found"}, status=HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:  # noqa: N802 - stdlib handler API.
        if urlparse(self.path).path != "/api/run":
            self._send_json({"error": "not found"}, status=HTTPStatus.NOT_FOUND)
            return
        try:
            filename, content, render_frames, compare_retargeters, source_human_height_m = self._read_motion_upload()
            if not filename:
                self._send_json({"error": "missing motion file"}, status=HTTPStatus.BAD_REQUEST)
                return
            if not content:
                self._send_json({"error": "empty motion file"}, status=HTTPStatus.BAD_REQUEST)
                return
            result = run_web_pipeline(
                content,
                filename,
                output_root=self.server.output_root,  # type: ignore[attr-defined]
                model_xml=self.server.model_xml,  # type: ignore[attr-defined]
                render_frames=render_frames,
                compare_retargeters=compare_retargeters,
                source_human_height_m=source_human_height_m,
            )
            self._send_json(result.to_dict())
        except Exception as exc:  # pragma: no cover - keeps web response debuggable.
            self._send_json(
                {"error": f"pipeline failed: {exc}"},
                status=HTTPStatus.INTERNAL_SERVER_ERROR,
            )

    def log_message(self, format: str, *args: object) -> None:
        print(f"{self.address_string()} - {format % args}")

    def _read_motion_upload(self) -> tuple[str, bytes, bool, bool, float | None]:
        content_type = self.headers.get("Content-Type", "")
        content_length = int(self.headers.get("Content-Length", "0") or "0")
        body = self.rfile.read(content_length)
        if "multipart/form-data" not in content_type:
            return "", b"", False, False, None
        message = BytesParser(policy=default).parsebytes(
            (
                f"Content-Type: {content_type}\r\n"
                "MIME-Version: 1.0\r\n"
                "\r\n"
            ).encode("utf-8")
            + body
        )
        filename = ""
        payload = b""
        render_frames = False
        compare_retargeters = False
        source_human_height_m: float | None = None
        for part in message.iter_parts():
            name = part.get_param("name", header="content-disposition")
            if name == "motion":
                filename = part.get_filename() or ""
                payload = part.get_payload(decode=True) or b""
                continue
            if name == "render_frames":
                value = (part.get_payload(decode=True) or b"").decode("utf-8", errors="ignore")
                render_frames = value.strip().lower() in {"1", "true", "yes", "on"}
            if name == "compare_retargeters":
                value = (part.get_payload(decode=True) or b"").decode("utf-8", errors="ignore")
                compare_retargeters = value.strip().lower() in {"1", "true", "yes", "on"}
            if name == "source_height_m":
                value = (part.get_payload(decode=True) or b"").decode("utf-8", errors="ignore").strip()
                if value:
                    try:
                        parsed = float(value)
                    except ValueError:
                        parsed = 0.0
                    if 0.5 <= parsed <= 2.5:
                        source_human_height_m = parsed
        return filename, payload, render_frames, compare_retargeters, source_human_height_m

    def _send_file(self, path: Path, content_type: str) -> None:
        if not path.exists():
            self._send_json({"error": "not found"}, status=HTTPStatus.NOT_FOUND)
            return
        data = path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_result(self, run_id: str) -> None:
        if not run_id or "/" in run_id or ".." in run_id:
            self._send_json({"error": "invalid run_id"}, status=HTTPStatus.BAD_REQUEST)
            return
        result_path = self.server.output_root / run_id / "pipeline_result.json"  # type: ignore[attr-defined]
        if not result_path.exists():
            self._send_json({"error": "result not found"}, status=HTTPStatus.NOT_FOUND)
            return
        self._send_json(json.loads(result_path.read_text(encoding="utf-8")))

    def _send_artifact(self, run_id: str, artifact_name: str) -> None:
        if not run_id or "/" in run_id or ".." in run_id:
            self._send_json({"error": "invalid run_id"}, status=HTTPStatus.BAD_REQUEST)
            return
        if not artifact_name or "/" in artifact_name or ".." in artifact_name:
            self._send_json({"error": "invalid artifact name"}, status=HTTPStatus.BAD_REQUEST)
            return
        run_dir = self.server.output_root / run_id  # type: ignore[attr-defined]
        result_path = run_dir / "pipeline_result.json"
        if not result_path.exists():
            self._send_json({"error": "result not found"}, status=HTTPStatus.NOT_FOUND)
            return
        result = json.loads(result_path.read_text(encoding="utf-8"))
        artifacts = result.get("artifacts", {})
        if not isinstance(artifacts, dict) or artifact_name not in artifacts:
            self._send_json({"error": "artifact not found"}, status=HTTPStatus.NOT_FOUND)
            return
        artifact_path = Path(str(artifacts[artifact_name])).resolve()
        run_root = run_dir.resolve()
        if run_root not in artifact_path.parents and artifact_path != run_root:
            self._send_json({"error": "artifact outside run directory"}, status=HTTPStatus.FORBIDDEN)
            return
        if not artifact_path.exists() or not artifact_path.is_file():
            self._send_json({"error": "artifact file not found"}, status=HTTPStatus.NOT_FOUND)
            return
        content_type = "video/mp4" if artifact_path.suffix.lower() == ".mp4" else "application/octet-stream"
        self._send_file(artifact_path, content_type)

    def _send_json(self, payload: object, status: HTTPStatus = HTTPStatus.OK) -> None:
        data = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="online-retarget-web")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--output-root", type=Path, default=WEB_RUN_ROOT)
    parser.add_argument("--model-xml", type=Path, default=DEFAULT_G1_MJCF)
    args = parser.parse_args(argv)

    args.output_root.mkdir(parents=True, exist_ok=True)
    server = ThreadingHTTPServer((args.host, args.port), RetargetWebHandler)
    server.output_root = args.output_root  # type: ignore[attr-defined]
    server.model_xml = args.model_xml  # type: ignore[attr-defined]
    url = f"http://{args.host}:{args.port}"
    print(f"Serving OnlineRetarget web console at {url}")
    print(f"Writing web runs under {args.output_root}")
    print(f"Using G1 MJCF {args.model_xml}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping OnlineRetarget web console")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
