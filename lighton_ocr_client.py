#!/usr/bin/env python3
"""
lighton_ocr_client.py
Client tipis untuk memanggil LightOnOCR-3 yang dijalankan lewat llama-server (llama.cpp).
Prasyarat: llama-server sudah berjalan di background (lihat run_ocr_server.bat).
"""

import base64
import os
from typing import Union
from io import BytesIO

import requests
from PIL import Image

# =====================================================================
# Konfigurasi endpoint llama-server
# =====================================================================
SERVER_HOST = os.environ.get("LIGHTON_OCR_HOST", "127.0.0.1")
SERVER_PORT = os.environ.get("LIGHTON_OCR_PORT", "8081")
SERVER_URL = f"http://{SERVER_HOST}:{SERVER_PORT}/v1/chat/completions"
HEALTH_URL = f"http://{SERVER_HOST}:{SERVER_PORT}/health"

DEFAULT_TIMEOUT = 1500  # detik
DEFAULT_MAX_TOKENS = 4096


class ServerNotReadyError(RuntimeError):
    """Dilempar kalau llama-server tidak bisa dihubungi / belum siap."""


class OcrRequestError(RuntimeError):
    """Dilempar kalau request OCR ke server gagal atau responsnya tidak valid."""


def is_server_ready(timeout: float = 3.0) -> bool:
    """Cek apakah llama-server sudah siap menerima request."""
    try:
        resp = requests.get(HEALTH_URL, timeout=timeout)
        return resp.status_code == 200
    except requests.exceptions.RequestException:
        return False


def _image_to_base64(image: Union[str, Image.Image]) -> str:
    if isinstance(image, str):
        if not os.path.exists(image):
            raise FileNotFoundError(f"File gambar tidak ditemukan: {image}")
        with open(image, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    elif isinstance(image, Image.Image):
        buf = BytesIO()
        image.convert("RGB").save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")
    else:
        raise TypeError("image harus berupa path (str) atau PIL.Image.Image")


def ocr_image(
    image: Union[str, Image.Image],
    timeout: float = DEFAULT_TIMEOUT,
) -> str:
    """
    Kirim satu gambar ke llama-server dan kembalikan teks hasil OCR.

    Args:
        image: path ke file gambar (str) atau objek PIL.Image.
        timeout: batas waktu tunggu response dalam detik (default 1500s).

    Raises:
        FileNotFoundError, ServerNotReadyError, OcrRequestError
    """
    image_b64 = _image_to_base64(image)

    payload = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image_b64}"},
                    }
                ],
            }
        ],
        "max_tokens": DEFAULT_MAX_TOKENS,
    }

    try:
        resp = requests.post(SERVER_URL, json=payload, timeout=timeout)
        resp.raise_for_status()
    except requests.exceptions.ConnectionError as e:
        raise ServerNotReadyError(
            f"Tidak bisa menghubungi llama-server di {SERVER_URL}. "
            f"Pastikan llama-server sudah dijalankan. Detail: {e}"
        ) from e
    except requests.exceptions.Timeout as e:
        raise OcrRequestError(f"Request OCR timeout setelah {timeout}s: {e}") from e
    except requests.exceptions.HTTPError as e:
        raise OcrRequestError(
            f"llama-server mengembalikan status {resp.status_code}: {resp.text[:500]}"
        ) from e

    try:
        data = resp.json()
        msg = data["choices"][0]["message"]
        text = msg.get("content") or msg.get("reasoning_content", "")
    except (KeyError, IndexError, ValueError, TypeError) as e:
        raise OcrRequestError(f"Format response tidak sesuai ekspektasi: {e}") from e

    return text.strip()


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Pemakaian: python lighton_ocr_client.py <path_gambar>")
        sys.exit(1)

    if not is_server_ready():
        print(f"❌ llama-server belum siap di {SERVER_URL}. Jalankan run_ocr_server.bat dulu.")
        sys.exit(1)

    target = sys.argv[1]
    try:
        result = ocr_image(target)
        with open("output.txt", "w", encoding="utf-8") as file:
            file.write(result)
        print(result)
    except (ServerNotReadyError, OcrRequestError, FileNotFoundError) as e:
        print(f"❌ {e}")
        sys.exit(1)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Pemakaian: python lighton_ocr_client.py <path_gambar>")
        sys.exit(1)

    if not is_server_ready():
        print(f"❌ llama-server belum siap di {SERVER_URL}. Jalankan run_ocr_server.bat dulu.")
        sys.exit(1)

    target = sys.argv[1]
    try:
        result = ocr_image(target)
        with open("output.txt", "w", encoding="utf-8") as file:
            file.write(result)
        print(result)
    except (ServerNotReadyError, OcrRequestError, FileNotFoundError) as e:
        print(f"❌ {e}")
        sys.exit(1)
