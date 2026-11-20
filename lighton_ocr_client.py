#!/usr/bin/env python3
"""
lighton_ocr_client.py
Client tipis untuk memanggil LightOnOCR-3-1B yang dijalankan lewat llama-server
(llama.cpp) dalam mode plain transcription.
Model GGUF: https://huggingface.co/richardyoung/LightOnOCR-3-1B-GGUF

Prasyarat: llama-server sudah jalan di background (lihat run_ocr_server.bat).
Modul ini HANYA mengirim HTTP request ke server tsb — tidak ada model
yang dimuat di proses Python.

Contoh pemakaian:

    from lighton_ocr_client import ocr_image, ServerNotReadyError

    # Plain transcription OCR (markdown text)
    text = ocr_image("image.png")
"""

import base64
import logging
import os
from typing import Union

import requests
from PIL import Image
from io import BytesIO

logger = logging.getLogger("lighton_ocr_client")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# =====================================================================
# Konfigurasi endpoint llama-server & model
# =====================================================================
SERVER_HOST = os.environ.get("LIGHTON_OCR_HOST", "127.0.0.1")
SERVER_PORT = os.environ.get("LIGHTON_OCR_PORT", "8081")
SERVER_URL = f"http://{SERVER_HOST}:{SERVER_PORT}/v1/chat/completions"
HEALTH_URL = f"http://{SERVER_HOST}:{SERVER_PORT}/health"
DEFAULT_MODEL = os.environ.get("LIGHTON_OCR_MODEL", "LightOnOCR-3-1B")

DEFAULT_TIMEOUT = 1200  # detik, sesuaikan kalau gambar besar / CPU lambat

# Parameter generasi default sesuai rekomendasi LightOnOCR-3
DEFAULT_TEMPERATURE = 0.0
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
    model: str = DEFAULT_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    timeout: float = DEFAULT_TIMEOUT,
) -> str:
    """
    Kirim satu gambar ke llama-server (LightOnOCR-3-1B) untuk plain transcription.

    Pada LightOnOCR-3, plain transcription mode dihasilkan saat gambar dikirim
    tanpa prompt teks tambahan, menghasilkan teks markdown dengan alur baca alami.

    Args:
        image: path ke file gambar, atau objek PIL.Image.
        model: nama identifier model (default: "LightOnOCR-3-1B").
        temperature: parameter temperature generasi (default: 0.0).
        max_tokens: batas token output maksimum (default: 4096).
        timeout: batas waktu tunggu response (detik).

    Raises:
        FileNotFoundError, ServerNotReadyError, OcrRequestError
    """
    image_b64 = _image_to_base64(image)

    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image_b64}"},
                    },
                    {
                        "type": "text",
                        "text": "",
                    },
                ],
            }
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    try:
        resp = requests.post(SERVER_URL, json=payload, timeout=timeout)
    except requests.exceptions.ConnectionError as e:
        raise ServerNotReadyError(
            f"Tidak bisa menghubungi llama-server di {SERVER_URL}. "
            f"Pastikan llama-server sudah dijalankan. Detail: {e}"
        ) from e
    except requests.exceptions.Timeout as e:
        raise OcrRequestError(f"Request OCR timeout setelah {timeout}s: {e}") from e

    if resp.status_code != 200:
        raise OcrRequestError(
            f"llama-server mengembalikan status {resp.status_code}: {resp.text[:500]}"
        )

    try:
        data = resp.json()
        text = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, ValueError) as e:
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
