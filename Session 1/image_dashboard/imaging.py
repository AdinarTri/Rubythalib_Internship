"""
imaging.py
Modul shared untuk logic image processing.
Dipakai bersama oleh fastapi_app.py dan flask_app.py agar tidak duplikasi kode.
"""

import cv2
import numpy as np


def read_image_from_bytes(data: bytes) -> np.ndarray:
    """Decode bytes upload menjadi array gambar OpenCV (BGR)."""
    if not data:
        raise ValueError("File kosong / tidak ada data yang diupload.")
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Gagal membaca gambar. Pastikan file yang diupload adalah gambar yang valid (jpg/png/bmp/dll).")
    return img


def encode_image_to_bytes(img: np.ndarray, ext: str = ".png") -> bytes:
    """Encode array gambar OpenCV menjadi bytes (default PNG)."""
    success, buffer = cv2.imencode(ext, img)
    if not success:
        raise ValueError("Gagal melakukan encode hasil gambar.")
    return buffer.tobytes()


def to_grayscale(img: np.ndarray) -> np.ndarray:
    """Konversi gambar ke grayscale (tetap 3 channel agar konsisten saat encode)."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def split_channels(img: np.ndarray) -> np.ndarray:
    """
    Pisahkan gambar menjadi channel Blue, Green, Red.
    Hasil: 1 gambar gabungan (3 channel disusun berdampingan/horizontal)
    supaya bisa dikembalikan sebagai satu file gambar.
    """
    b, g, r = cv2.split(img)
    zeros = np.zeros_like(b)

    blue_only = cv2.merge([b, zeros, zeros])
    green_only = cv2.merge([zeros, g, zeros])
    red_only = cv2.merge([zeros, zeros, r])

    combined = np.hstack([blue_only, green_only, red_only])
    return combined


def crop_image(img: np.ndarray, x: int, y: int, width: int, height: int) -> np.ndarray:
    """Crop gambar sesuai koordinat (x, y) dan ukuran (width, height)."""
    h, w = img.shape[:2]

    x = max(0, min(x, w - 1))
    y = max(0, min(y, h - 1))
    x2 = min(x + width, w)
    y2 = min(y + height, h)

    if x2 <= x or y2 <= y:
        raise ValueError(
            f"Koordinat crop tidak valid untuk ukuran gambar {w}x{h}. "
            f"Diminta x={x}, y={y}, width={width}, height={height}."
        )

    return img[y:y2, x:x2]


def detect_lines(
    img: np.ndarray,
    canny_low: int = 50,
    canny_high: int = 150,
    hough_threshold: int = 100,
) -> np.ndarray:
    """
    Deteksi garis menggunakan Canny edge detection + Probabilistic Hough Transform.
    Garis yang terdeteksi digambar berwarna merah di atas gambar asli.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, canny_low, canny_high)
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180,
        threshold=hough_threshold,
        minLineLength=50,
        maxLineGap=10,
    )

    result = img.copy()
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(result, (x1, y1), (x2, y2), (0, 0, 255), 2)

    return result
