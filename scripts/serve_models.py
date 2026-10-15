"""測試安裝程式用的本機模型伺服器：把已經下載好的模型檔，用 Hugging Face 的網址格式提供出來。

安裝程式從 ``<主機>/<repo>/resolve/<版本>/<檔名>`` 下載模型；用 ``build_installer.ps1 -ModelHost http://127.0.0.1:8765`` 建出
指向這台伺服器的測試安裝程式，就不必真的從網路下載好幾 GB。檔案取自 ``models/translation/<名稱>/``（先用
``fetch_translation_models.py`` 下載好）。

    python scripts/serve_models.py                      # 127.0.0.1:8765
    python scripts/serve_models.py --tamper qwen2.5-0.5b-instruct-gguf-q8_0   # 該模型的授權檔內容被改過，安裝程式應該拒絕安裝

只綁在本機位址；Ctrl+C 結束。
"""

from __future__ import annotations

import argparse
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import installer_models  # noqa: E402


def build_routes(catalog: dict, models_dir: Path) -> dict[str, tuple[Path, str]]:
    """網址路徑 → （檔案, 模型名稱）。"""
    routes: dict[str, tuple[Path, str]] = {}
    for model in catalog["models"]:
        for item in [*model["files"], model["license_file"]]:
            path = f"/{model['repo']}/resolve/{model['revision']}/{item['name']}"
            routes[path] = (models_dir / model["name"] / item["name"], model["name"])
    return routes


def make_handler(routes: dict, tamper: set[str]):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, head: bool) -> None:
            entry = routes.get(self.path.split("?", 1)[0])
            if entry is None or not entry[0].is_file():
                self.send_error(404)
                return
            path, name = entry
            data_size = path.stat().st_size
            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", str(data_size))
            self.end_headers()
            if head:
                return
            with open(path, "rb") as handle:
                first = True
                while chunk := handle.read(1 << 20):
                    if first and name in tamper and path.name == "LICENSE":
                        chunk = b"X" + chunk[1:]  # 同樣大小、內容不同：SHA-256 不會過
                    first = False
                    self.wfile.write(chunk)

        def do_GET(self):  # noqa: N802
            self._send(False)

        def do_HEAD(self):  # noqa: N802
            self._send(True)

        def log_message(self, fmt, *args):  # noqa: A003
            print(f"{self.address_string()} {fmt % args}", flush=True)

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--models-dir", type=Path, default=ROOT / "models" / "translation")
    parser.add_argument("--tamper", nargs="*", default=[], help="這些模型的授權檔回傳被改過的內容（測試安裝程式會不會拒絕）")
    args = parser.parse_args()
    routes = build_routes(installer_models.load(), args.models_dir)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(routes, set(args.tamper)))
    print(f"提供 {len(routes)} 個檔案：http://127.0.0.1:{args.port}/", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
