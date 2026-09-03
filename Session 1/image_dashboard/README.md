# Image Processing Dashboard (Flask + FastAPI)

Dashboard sederhana untuk 4 operasi image processing:
1. **Grayscale** — konversi gambar ke hitam-putih
2. **Split Channel** — pisahkan gambar menjadi channel Blue/Green/Red
3. **Cropping** — potong gambar sesuai koordinat & ukuran
4. **Line Detection** — deteksi garis (Canny edge detection + Hough Transform)

Dua implementasi backend disediakan, memakai logic processing yang sama (`imaging.py`):
- `fastapi_app.py` — backend FastAPI, dengan dokumentasi otomatis (Swagger/ReDoc)
- `flask_app.py` — backend Flask, dengan halaman dokumentasi manual di `/api-docs`

## Struktur folder

```
image_dashboard/
├── imaging.py           # logic image processing (dipakai bersama)
├── fastapi_app.py        # server FastAPI
├── flask_app.py           # server Flask
├── requirements.txt
├── static/
│   ├── style.css          # dipakai bersama (Flask & FastAPI)
│   ├── dashboard.js        # dipakai bersama (Flask & FastAPI)
│   └── index.html          # dashboard untuk FastAPI (disajikan langsung sebagai HTML)
└── templates/
    ├── index.html          # dashboard untuk Flask (Jinja template)
    └── api_docs.html         # halaman dokumentasi API untuk Flask
```

## Instalasi

```bash
python -m venv venv
source venv/bin/activate     # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Menjalankan FastAPI

```bash
uvicorn fastapi_app:app --reload --port 8000
```

- Dashboard: http://127.0.0.1:8000/
- Swagger UI (dokumentasi interaktif otomatis): http://127.0.0.1:8000/docs
- ReDoc: http://127.0.0.1:8000/redoc

## Menjalankan Flask

```bash
python flask_app.py
```

- Dashboard: http://127.0.0.1:5000/
- Dokumentasi endpoint: http://127.0.0.1:5000/api-docs

> Kedua server bisa dijalankan bersamaan (port berbeda: 8000 vs 5000) karena tidak saling bergantung.

## Dokumentasi Endpoint

Kedua backend (Flask & FastAPI) memakai **path dan parameter yang sama**, hanya berbeda base URL/port.

### `POST /api/grayscale`
Konversi gambar ke grayscale.

| Field | Lokasi | Wajib | Keterangan |
|---|---|---|---|
| `file` | form-data | Ya | File gambar (jpg/png/bmp) |

Response: `image/png`.

### `POST /api/split-channels`
Pisahkan gambar menjadi channel Blue, Green, Red — hasil digabung berdampingan (horizontal) dalam satu gambar output.

| Field | Lokasi | Wajib | Keterangan |
|---|---|---|---|
| `file` | form-data | Ya | File gambar |

Response: `image/png` (lebar = 3x lebar gambar asli).

### `POST /api/crop`
Potong gambar sesuai koordinat.

| Field | Lokasi | Wajib | Default | Keterangan |
|---|---|---|---|---|
| `file` | form-data | Ya | — | File gambar |
| `x` | query/form | Tidak | 0 | Koordinat X awal crop |
| `y` | query/form | Tidak | 0 | Koordinat Y awal crop |
| `width` | query/form | Tidak | 100 | Lebar area crop |
| `height` | query/form | Tidak | 100 | Tinggi area crop |

Contoh (FastAPI):
```
POST http://127.0.0.1:8000/api/crop?x=10&y=10&width=300&height=200
```

### `POST /api/line-detection`
Deteksi garis menggunakan Canny edge detection + Probabilistic Hough Transform. Garis digambar merah di atas gambar asli.

| Field | Lokasi | Wajib | Default | Keterangan |
|---|---|---|---|---|
| `file` | form-data | Ya | — | File gambar |
| `canny_low` | query/form | Tidak | 50 | Threshold bawah Canny |
| `canny_high` | query/form | Tidak | 150 | Threshold atas Canny |
| `hough_threshold` | query/form | Tidak | 100 | Minimal vote Hough Transform |

## Contoh pemakaian dengan curl

```bash
# Grayscale
curl -X POST -F "file=@foto.jpg" http://127.0.0.1:8000/api/grayscale -o hasil_gray.png

# Split channel
curl -X POST -F "file=@foto.jpg" http://127.0.0.1:8000/api/split-channels -o hasil_split.png

# Crop
curl -X POST -F "file=@foto.jpg" "http://127.0.0.1:8000/api/crop?x=0&y=0&width=300&height=300" -o hasil_crop.png

# Line detection
curl -X POST -F "file=@foto.jpg" "http://127.0.0.1:8000/api/line-detection?canny_low=50&canny_high=150" -o hasil_line.png
```

Ganti port `8000` menjadi `5000` untuk menguji versi Flask.

## Catatan teknis

- Semua endpoint menerima 1 file gambar (`file`) dan mengembalikan gambar hasil sebagai `image/png`.
- Error (misalnya file bukan gambar valid, atau koordinat crop di luar batas) dikembalikan dengan status `400` dan body JSON `{"detail": "..."}` (FastAPI) atau `{"error": "..."}` (Flask).
- Logic image processing terpusat di `imaging.py` sehingga kedua server 100% konsisten hasilnya.
