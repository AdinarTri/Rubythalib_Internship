"""
fastapi_app.py
Dashboard image processing sederhana menggunakan FastAPI.

Menjalankan:
    uvicorn fastapi_app:app --reload --port 8000

Dashboard  : http://127.0.0.1:8000/
Swagger UI : http://127.0.0.1:8000/docs
ReDoc      : http://127.0.0.1:8000/redoc
"""

import io

from fastapi import FastAPI, UploadFile, File, Query, HTTPException
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from imaging import (
    read_image_from_bytes,
    encode_image_to_bytes,
    to_grayscale,
    split_channels,
    crop_image,
    detect_lines,
)

app = FastAPI(
    title="Image Processing Dashboard API (FastAPI)",
    description=(
        "API sederhana untuk image processing: grayscale, split channel, "
        "cropping, dan line detection. Upload gambar lalu terima hasil "
        "gambar yang sudah diproses."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")


def _run_and_stream(data: bytes, processor, **kwargs) -> StreamingResponse:
    try:
        img = read_image_from_bytes(data)
        result = processor(img, **kwargs)
        out_bytes = encode_image_to_bytes(result)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return StreamingResponse(io.BytesIO(out_bytes), media_type="image/png")


@app.get(
    "/",
    response_class=HTMLResponse,
    summary="Dashboard utama",
    description="Halaman web untuk upload gambar dan mencoba semua fitur image processing secara interaktif.",
    include_in_schema=False,
)
async def dashboard():
    with open("static/index.html", "r", encoding="utf-8") as f:
        return f.read()


@app.post(
    "/api/grayscale",
    summary="Konversi ke grayscale",
    description="Upload sebuah gambar, dan dapatkan versi grayscale (hitam-putih) dari gambar tersebut.",
    responses={200: {"content": {"image/png": {}}}, 400: {"description": "Gambar tidak valid"}},
)
async def grayscale_endpoint(
    file: UploadFile = File(..., description="File gambar yang akan diproses (jpg, png, bmp, dll).")
):
    data = await file.read()
    return _run_and_stream(data, to_grayscale)


@app.post(
    "/api/split-channels",
    summary="Split channel RGB",
    description=(
        "Upload sebuah gambar. Hasilnya berupa satu gambar berisi 3 channel "
        "warna (Blue, Green, Red) yang disusun berdampingan secara horizontal."
    ),
    responses={200: {"content": {"image/png": {}}}, 400: {"description": "Gambar tidak valid"}},
)
async def split_channels_endpoint(
    file: UploadFile = File(..., description="File gambar yang akan diproses (jpg, png, bmp, dll).")
):
    data = await file.read()
    return _run_and_stream(data, split_channels)


@app.post(
    "/api/crop",
    summary="Crop gambar",
    description="Upload gambar dan potong (crop) sesuai koordinat x, y serta ukuran width, height yang diberikan.",
    responses={200: {"content": {"image/png": {}}}, 400: {"description": "Koordinat crop tidak valid"}},
)
async def crop_endpoint(
    file: UploadFile = File(..., description="File gambar yang akan diproses."),
    x: int = Query(0, ge=0, description="Koordinat X titik awal crop (pixel)."),
    y: int = Query(0, ge=0, description="Koordinat Y titik awal crop (pixel)."),
    width: int = Query(100, gt=0, description="Lebar area crop (pixel)."),
    height: int = Query(100, gt=0, description="Tinggi area crop (pixel)."),
):
    data = await file.read()
    return _run_and_stream(data, crop_image, x=x, y=y, width=width, height=height)


@app.post(
    "/api/line-detection",
    summary="Deteksi garis (line detection)",
    description=(
        "Upload gambar. Sistem mendeteksi garis menggunakan Canny edge "
        "detection lalu Probabilistic Hough Transform. Garis yang "
        "terdeteksi digambar berwarna merah di atas gambar asli."
    ),
    responses={200: {"content": {"image/png": {}}}, 400: {"description": "Gambar tidak valid"}},
)
async def line_detection_endpoint(
    file: UploadFile = File(..., description="File gambar yang akan diproses."),
    canny_low: int = Query(50, ge=0, description="Threshold bawah untuk Canny edge detector."),
    canny_high: int = Query(150, ge=0, description="Threshold atas untuk Canny edge detector."),
    hough_threshold: int = Query(100, ge=1, description="Minimal vote agar sebuah garis dianggap terdeteksi (Hough Transform)."),
):
    data = await file.read()
    return _run_and_stream(
        data, detect_lines,
        canny_low=canny_low, canny_high=canny_high, hough_threshold=hough_threshold,
    )
