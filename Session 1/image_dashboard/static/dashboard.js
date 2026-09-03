// shared_dashboard.js
// Logic dashboard untuk semua tab. Endpoint dipanggil relatif (/api/...)
// sehingga file ini bisa dipakai baik oleh server Flask maupun FastAPI.

function showTab(tabId) {
  document.querySelectorAll(".panel").forEach((p) => p.classList.remove("active"));
  document.querySelectorAll(".tab-btn").forEach((b) => b.classList.remove("active"));
  document.getElementById(tabId).classList.add("active");
  document.querySelector(`.tab-btn[data-tab="${tabId}"]`).classList.add("active");
}

function previewOriginal(inputEl, imgElId) {
  const file = inputEl.files[0];
  if (!file) return;
  const url = URL.createObjectURL(file);
  document.getElementById(imgElId).src = url;
}

async function runProcess({ endpoint, fileInputId, params, resultImgId, statusId, button }) {
  const fileInput = document.getElementById(fileInputId);
  const statusEl = document.getElementById(statusId);
  const resultImg = document.getElementById(resultImgId);

  if (!fileInput.files[0]) {
    statusEl.textContent = "Pilih file gambar terlebih dahulu.";
    statusEl.className = "status error";
    return;
  }

  const formData = new FormData();
  formData.append("file", fileInput.files[0]);

  let url = endpoint;
  if (params && Object.keys(params).length > 0) {
    const qs = new URLSearchParams(params).toString();
    url = `${endpoint}?${qs}`;
  }

  button.disabled = true;
  statusEl.textContent = "Memproses...";
  statusEl.className = "status";

  try {
    const res = await fetch(url, { method: "POST", body: formData });
    if (!res.ok) {
      let msg = `Gagal (status ${res.status})`;
      try {
        const errJson = await res.json();
        if (errJson.detail) msg = errJson.detail;
        if (errJson.error) msg = errJson.error;
      } catch (_) {}
      throw new Error(msg);
    }
    const blob = await res.blob();
    resultImg.src = URL.createObjectURL(blob);
    statusEl.textContent = "Berhasil diproses.";
    statusEl.className = "status ok";
  } catch (err) {
    statusEl.textContent = err.message || "Terjadi kesalahan.";
    statusEl.className = "status error";
  } finally {
    button.disabled = false;
  }
}

document.addEventListener("DOMContentLoaded", () => {
  // Grayscale
  document.getElementById("gray-file").addEventListener("change", (e) => previewOriginal(e.target, "gray-original"));
  document.getElementById("gray-run").addEventListener("click", () => {
    runProcess({
      endpoint: "/api/grayscale",
      fileInputId: "gray-file",
      params: {},
      resultImgId: "gray-result",
      statusId: "gray-status",
      button: document.getElementById("gray-run"),
    });
  });

  // Split channels
  document.getElementById("split-file").addEventListener("change", (e) => previewOriginal(e.target, "split-original"));
  document.getElementById("split-run").addEventListener("click", () => {
    runProcess({
      endpoint: "/api/split-channels",
      fileInputId: "split-file",
      params: {},
      resultImgId: "split-result",
      statusId: "split-status",
      button: document.getElementById("split-run"),
    });
  });

  // Crop
  document.getElementById("crop-file").addEventListener("change", (e) => previewOriginal(e.target, "crop-original"));
  document.getElementById("crop-run").addEventListener("click", () => {
    runProcess({
      endpoint: "/api/crop",
      fileInputId: "crop-file",
      params: {
        x: document.getElementById("crop-x").value,
        y: document.getElementById("crop-y").value,
        width: document.getElementById("crop-width").value,
        height: document.getElementById("crop-height").value,
      },
      resultImgId: "crop-result",
      statusId: "crop-status",
      button: document.getElementById("crop-run"),
    });
  });

  // Line detection
  document.getElementById("line-file").addEventListener("change", (e) => previewOriginal(e.target, "line-original"));
  document.getElementById("line-run").addEventListener("click", () => {
    runProcess({
      endpoint: "/api/line-detection",
      fileInputId: "line-file",
      params: {
        canny_low: document.getElementById("line-canny-low").value,
        canny_high: document.getElementById("line-canny-high").value,
        hough_threshold: document.getElementById("line-hough").value,
      },
      resultImgId: "line-result",
      statusId: "line-status",
      button: document.getElementById("line-run"),
    });
  });
});
