"""
flask_app.py
Dashboard image processing sederhana menggunakan Flask.

Menjalankan:
    python flask_app.py

Dashboard        : http://127.0.0.1:5000/
Dokumentasi API  : http://127.0.0.1:5000/api-docs
"""

import io

from flask import Flask, request, send_file, render_template, jsonify

from imaging import (
    read_image_from_bytes,
    encode_image_to_bytes,
    to_grayscale,
    split_channels,
    crop_image,
    detect_lines,
)

app = Flask(__name__)


def _get_uploaded_image():
    if "file" not in request.files:
        raise ValueError("Tidak ada file yang diupload. Gunakan field form-data dengan nama 'file'.")
    f = request.files["file"]
    data = f.read()
    return read_image_from_bytes(data)


def _get_int_param(name: str, default: int) -> int:
    raw = request.args.get(name, request.form.get(name, default))
    try:
        return int(raw)
    except (TypeError, ValueError):
        raise ValueError(f"Parameter '{name}' harus berupa angka bulat, diterima: {raw!r}")


def _respond_image(img):
    out_bytes = encode_image_to_bytes(img)
    return send_file(io.BytesIO(out_bytes), mimetype="image/png")


@app.route("/")
def dashboard():
    return render_template("index.html")


@app.route("/api-docs")
def api_docs():
    return render_template("api_docs.html")


@app.route("/api/grayscale", methods=["POST"])
def grayscale_route():
    """Upload gambar -> kembalikan versi grayscale."""
    try:
        img = _get_uploaded_image()
        result = to_grayscale(img)
        return _respond_image(result)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400


@app.route("/api/split-channels", methods=["POST"])
def split_channels_route():
    """Upload gambar -> kembalikan gabungan channel Blue/Green/Red berdampingan."""
    try:
        img = _get_uploaded_image()
        result = split_channels(img)
        return _respond_image(result)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400


@app.route("/api/crop", methods=["POST"])
def crop_route():
    """Upload gambar + query/form params x,y,width,height -> kembalikan hasil crop."""
    try:
        img = _get_uploaded_image()
        x = _get_int_param("x", 0)
        y = _get_int_param("y", 0)
        width = _get_int_param("width", 100)
        height = _get_int_param("height", 100)
        result = crop_image(img, x, y, width, height)
        return _respond_image(result)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400


@app.route("/api/line-detection", methods=["POST"])
def line_detection_route():
    """Upload gambar + query/form params canny_low, canny_high, hough_threshold -> kembalikan hasil deteksi garis."""
    try:
        img = _get_uploaded_image()
        canny_low = _get_int_param("canny_low", 50)
        canny_high = _get_int_param("canny_high", 150)
        hough_threshold = _get_int_param("hough_threshold", 100)
        result = detect_lines(img, canny_low, canny_high, hough_threshold)
        return _respond_image(result)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(debug=True, port=5000)
