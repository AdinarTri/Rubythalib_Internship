"""
BDC Satria Data 2026 - reusable training pipeline (EfficientNet family & friends).

Semua eksperimen digerakkan oleh satu dict CONFIG. Untuk utak-atik parameter:
edit CONFIG (di notebook) lalu panggil run(cfg). Tidak perlu sentuh fungsi di bawah.

Menghasilkan seluruh deliverable guideline ke experiments/<name>/:
  <name>.keras, history.csv, classification_report.txt, confusion_matrix.png,
  training_curve.png, loss_curve.png, parameter.json, README.md, notes.md

Metrik penilaian kompetisi = Macro F1. Checkpoint & EarlyStopping dipantau ke val macro-F1.
"""

import json, os, time, random
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score)
from sklearn.utils.class_weight import compute_class_weight

__version__ = "21-dino-cls"         # naikkan tiap edit; run() cetak ini biar tahu versi mana yg jalan

# ---- label kompetisi: 0=Recyclable, 1=Electronic, 2=Organic (dari prefix folder) ----
CLASS_NAMES = ["Recyclable", "Electronic", "Organic"]

try:  # izinkan load model berisi Lambda (preproc caffe ResNet); no-op di keras lama
    tf.keras.config.enable_unsafe_deserialization()
except AttributeError:
    pass

# ---- registry backbone: nama -> (konstruktor keras.applications) ----
# EfficientNet keras sudah normalisasi input di dalam model -> beri piksel [0,255], JANGAN rescale.
def _backbones():
    from tensorflow.keras import applications as A
    return {
        "EfficientNetB0":   A.EfficientNetB0,
        "EfficientNetV2B0": A.EfficientNetV2B0,
        "EfficientNetV2S":  A.EfficientNetV2S,
        "EfficientNetV2L":  A.EfficientNetV2L,
        # tambahan anggota tim lain kalau perlu pakai template yang sama:
        "ResNet50":         A.ResNet50,
        "ResNet101":        A.ResNet101,
        "DenseNet121":      A.DenseNet121,
        "DenseNet169":      A.DenseNet169,
        "ConvNeXtTiny":     getattr(A, "ConvNeXtTiny", None),
        "MobileNetV3Large": A.MobileNetV3Large,
    }


# EfficientNet/V2, ConvNeXt, MobileNetV3: normalisasi built-in di dalam model -> input [0,255].
# ResNet & DenseNet TIDAK: wajib preprocessing eksplisit, ditanam di dalam model agar
# ikut tersimpan di .keras (ensemble/predict tak perlu tahu apa-apa).
_PREPROC = {"ResNet50": "caffe", "ResNet101": "caffe",
            "DenseNet121": "torch", "DenseNet169": "torch"}

# ---- default config (Tahap 1 guideline). Override lewat notebook. ----
CONFIG = {
    "backbone": "EfficientNetB0",
    "img_size": 224,
    "batch_size": 32,
    "epochs": 30,
    "lr": 1e-4,
    "optimizer": "adamw",            # adam | adamw | sgd
    "weight_decay": 1e-4,
    "label_smoothing": 0.0,
    "dropout": 0.2,
    "val_split": 0.15,
    "seed": 42,
    "train_dir": "train",
    "test_dir": "test",
    "out_root": "experiments",
    # fine tuning: "freeze" | "last20" | "last50" | "full"
    "finetune": "freeze",
    "use_class_weight": True,        # penting utk Electronic (kelas minoritas)
    "augment": True,                 # standard: flip + rotation + zoom
    "mixed_precision": False,        # True di GPU (wajib untuk RTX 3050 4GB)
    "es_patience": 6,
    "rlrop_patience": 3,
    "verbose": 1,                    # progress bar Keras
    "log_every": 50,                 # heartbeat: cetak tiap N step (dijamin ke-flush di Colab)
    "make_submission": False,        # True -> tulis submission.csv dari folder test
    "smoke": False,                  # True -> subset kecil + 1 epoch (uji end-to-end)
    "cv_folds": 0,                   # >0 -> pakai StratifiedKFold (lihat run_cv)
    "fold": 0,                       # fold aktif saat cv_folds > 0
    "warmup_epochs": 0,              # >0 -> latih head dulu (backbone beku, lr 1e-3) sebelum unfreeze
    "random_erase": 0.0,             # probabilitas Cutout/Random Erasing per gambar (0 = off, coba 0.25)
}


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); tf.random.set_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def list_samples(train_dir):
    """Kumpulkan (filepath, label_int) dari subfolder berprefix '0_','1_','2_'."""
    paths, labels = [], []
    for d in sorted(os.listdir(train_dir)):
        sub = os.path.join(train_dir, d)
        if not os.path.isdir(sub):
            continue
        y = int(d.split("_")[0])          # prefix = kode label kompetisi
        for f in sorted(os.listdir(sub)):   # WAJIB sorted: urutan listdir beda antar sesi ->
            paths.append(os.path.join(sub, f)); labels.append(y)   # split CV tidak reproducible
    return np.array(paths), np.array(labels)


def make_ds(paths, labels, size, batch, training, augment, seed, erase_p=0.0):
    """tf.data: baca dari disk (streaming, hemat RAM), decode 3D-aman, resize, augment (train saja)."""
    if training:
        # acak urutan array SEBELUM masuk tf.data: StratifiedKFold mengembalikan indeks
        # terurut per kelas, dan buffer shuffle 4096 << dataset -> batch awal epoch nyaris
        # satu kelas semua (loss meledak tiap awal epoch). Permutasi penuh menyembuhkannya.
        perm = np.random.RandomState(seed).permutation(len(paths))
        paths, labels = paths[perm], labels[perm]

    def load(path, y):
        # decode_image + expand_animations=False -> dijamin [H,W,3] utk jpg/png/gif/bmp
        img = tf.io.decode_image(tf.io.read_file(path), channels=3, expand_animations=False)
        if training and augment:
            # semua via tf.image (rank-safe, tanpa layer resize yang dulu bikin error 5D)
            img = tf.image.resize(img, [size + 32, size + 32])
            img = tf.image.random_crop(img, [size, size, 3], seed=seed)   # ~ random zoom/translate
            img = tf.image.random_flip_left_right(img)
            img = tf.image.random_brightness(img, 40.0)                   # skala piksel [0,255]
            img = tf.image.random_contrast(img, 0.8, 1.2)
            # anti bias warna->Organic: goyang warna + kadang grayscale (belajar bentuk/material)
            img = tf.image.random_saturation(img, 0.6, 1.4)
            img = tf.image.random_hue(img, 0.05)
            img = tf.cond(tf.random.uniform([]) < 0.1,
                          lambda: tf.image.grayscale_to_rgb(tf.image.rgb_to_grayscale(img)),
                          lambda: img)
            if erase_p > 0:
                # Random Erasing: hapus kotak acak (1/8-1/4 sisi) -> model dipaksa pakai
                # BANYAK ciri penyusun objek, bukan satu tekstur andalan yang mirip antar kelas
                def _erase(im):
                    eh = tf.random.uniform([], size // 8, size // 4, tf.int32)
                    ew = tf.random.uniform([], size // 8, size // 4, tf.int32)
                    y0 = tf.random.uniform([], 0, size - eh, tf.int32)
                    x0 = tf.random.uniform([], 0, size - ew, tf.int32)
                    mask = tf.pad(tf.zeros([eh, ew, 3]),
                                  [[y0, size - eh - y0], [x0, size - ew - x0], [0, 0]],
                                  constant_values=1.0)
                    return im * mask + 127.0 * (1.0 - mask)   # kotak abu-abu netral
                img = tf.cond(tf.random.uniform([]) < erase_p, lambda: _erase(img), lambda: img)
            img = tf.clip_by_value(img, 0.0, 255.0)
        else:
            img = tf.image.resize(img, [size, size])
        img = tf.ensure_shape(img, [size, size, 3])     # pagar: paksa rank benar
        img = tf.cast(img, tf.float32)                  # [0,255], normalisasi ada di dalam EfficientNet
        return img, y

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.map(load, num_parallel_calls=tf.data.AUTOTUNE)
    if training:
        ds = ds.shuffle(min(len(paths), 4096), seed=seed, reshuffle_each_iteration=True)
    return ds.batch(batch).prefetch(tf.data.AUTOTUNE)


def _apply_finetune(base, ft):
    if ft == "freeze":
        base.trainable = False
    elif ft == "full":
        base.trainable = True
    else:  # lastN
        n = int(ft.replace("last", ""))
        base.trainable = True
        for layer in base.layers[:-n]:
            layer.trainable = False


def build_model(cfg):
    if "vit" in cfg["backbone"]:      # vit_* dan dinov3_vit_* (preset keras_hub)
        # Transformer via keras_hub (pip install keras-hub). Nama backbone = nama preset,
        # mis. "vit_base_patch16_384_imagenet". Preprocessing bawaan preset (auto).
        import keras_hub
        try:
            model = keras_hub.models.ImageClassifier.from_preset(
                cfg["backbone"], num_classes=len(CLASS_NAMES), activation="softmax")
            _apply_finetune(model.backbone, cfg["finetune"])
            return model
        except Exception as e1:
            # fallback utk preset backbone-only (SSL spt DINOv3: tanpa classifier head):
            # Backbone + normalisasi ImageNet + pooling + head sendiri
            try:
                bb = keras_hub.models.Backbone.from_preset(cfg["backbone"])
            except Exception as e2:
                avail = []
                for attr in ("ViTImageClassifier", "ImageClassifier", "Backbone"):
                    klass = getattr(keras_hub.models, attr, None)
                    if klass is not None and hasattr(klass, "presets"):
                        avail += [p for p in klass.presets if "vit" in p]
                raise RuntimeError(
                    f"Preset '{cfg['backbone']}' gagal.\nImageClassifier: {e1}\nBackbone: {e2}\n"
                    f"Preset tersedia: {sorted(set(avail))}") from e2
            size = cfg["img_size"]
            inp = tf.keras.Input(shape=(size, size, 3))
            x = tf.keras.layers.Rescaling(1 / 255.0)(inp)          # DINO: standarisasi ImageNet
            x = tf.keras.layers.Normalization(
                mean=[0.485, 0.456, 0.406],
                variance=[0.229 ** 2, 0.224 ** 2, 0.225 ** 2])(x)
            x = bb({"pixel_values": x})                            # DINOv3 minta input dict
            if isinstance(x, dict):
                x = x.get("last_hidden_state", list(x.values())[-1])
            if len(x.shape) == 3:
                # [B, token, dim]: token 0 = CLS. JANGAN GAP semua token (ada register
                # token DINOv3 yg mengencerkan sinyal -> fitur nyaris konstan). Ambil CLS.
                x = tf.keras.layers.Lambda(lambda t: t[:, 0], name="cls_token")(x)
            elif len(x.shape) == 4:                                # [B, H, W, C]
                x = tf.keras.layers.GlobalAveragePooling2D()(x)
            x = tf.keras.layers.Dropout(cfg["dropout"])(x)
            out = tf.keras.layers.Dense(len(CLASS_NAMES), activation="softmax",
                                        dtype="float32")(x)
            model = tf.keras.Model(inp, out)
            model.backbone = bb                                     # utk warmup & finetune
            _apply_finetune(bb, cfg["finetune"])
            print(f"[build_model] fallback Backbone+head utk {cfg['backbone']}", flush=True)
            return model

    ctor = _backbones()[cfg["backbone"]]
    if ctor is None:
        raise ValueError(f"{cfg['backbone']} tidak tersedia di versi keras ini")
    size = cfg["img_size"]
    base = ctor(include_top=False, weights="imagenet",
                input_shape=(size, size, 3), pooling="avg")
    _apply_finetune(base, cfg["finetune"])

    inp = tf.keras.Input(shape=(size, size, 3))
    x = inp
    mode = _PREPROC.get(cfg["backbone"])
    if mode == "caffe":    # ResNet keras: RGB->BGR + kurangi mean ImageNet
        x = tf.keras.layers.Lambda(
            lambda t: t[..., ::-1] - tf.constant([103.939, 116.779, 123.68]),
            name="preproc_caffe")(x)
    elif mode == "torch":  # DenseNet keras: skala [0,1] lalu standarisasi ImageNet
        x = tf.keras.layers.Rescaling(1 / 255.0)(x)
        x = tf.keras.layers.Normalization(
            mean=[0.485, 0.456, 0.406],
            variance=[0.229 ** 2, 0.224 ** 2, 0.225 ** 2])(x)
    x = base(x)    # training flag dinamis: BN mode train saat fit, mode inference saat predict/val
    x = tf.keras.layers.Dropout(cfg["dropout"])(x)
    out = tf.keras.layers.Dense(len(CLASS_NAMES), activation="softmax", dtype="float32")(x)
    return tf.keras.Model(inp, out)


def make_optimizer(cfg):
    lr, wd = cfg["lr"], cfg["weight_decay"]
    opt = cfg["optimizer"].lower()
    if opt == "adam":
        return tf.keras.optimizers.Adam(lr)
    if opt == "adamw":
        return tf.keras.optimizers.AdamW(learning_rate=lr, weight_decay=wd)
    if opt == "sgd":
        return tf.keras.optimizers.SGD(lr, momentum=0.9)
    raise ValueError(opt)


class BatchLog(tf.keras.callbacks.Callback):
    """Heartbeat per-step: cetak progres tiap N batch (flush=True) biar kelihatan di Colab."""
    def __init__(self, every, steps):
        super().__init__(); self.every, self.steps, self.t0 = every, steps, None

    def on_epoch_begin(self, epoch, logs=None):
        self.t0 = time.time(); print(f"Epoch {epoch+1} mulai...", flush=True)

    def on_train_batch_end(self, batch, logs=None):
        if batch % self.every == 0:
            logs = logs or {}
            rate = (batch + 1) / max(time.time() - self.t0, 1e-6)
            print(f"  step {batch}/{self.steps}  loss={logs.get('loss', 0):.4f} "
                  f"acc={logs.get('accuracy', 0):.4f}  ({rate:.1f} step/s)", flush=True)


class MacroF1(tf.keras.callbacks.Callback):
    """Hitung val macro-F1 tiap epoch (metrik kompetisi) + simpan bobot terbaik."""
    def __init__(self, val_ds, ckpt_path, patience):
        super().__init__()
        self.val_ds, self.ckpt_path, self.patience = val_ds, ckpt_path, patience
        self.best, self.best_epoch, self.wait = -1.0, 0, 0

    def on_epoch_end(self, epoch, logs=None):
        y_true = np.concatenate([y.numpy() for _, y in self.val_ds])
        y_pred = self.model.predict(self.val_ds, verbose=0).argmax(1)
        f1 = f1_score(y_true, y_pred, average="macro")
        logs = logs or {}; logs["val_macro_f1"] = f1
        if f1 > self.best:
            self.best, self.best_epoch, self.wait = f1, epoch, 0
            self.model.save_weights(self.ckpt_path)
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.model.stop_training = True
        print(f"  val_macro_f1={f1:.4f} (best={self.best:.4f}@{self.best_epoch+1})")


def run(cfg=None):
    print(f"[pipeline v{__version__}]")
    cfg = {**CONFIG, **(cfg or {})}
    set_seed(cfg["seed"])
    if cfg["mixed_precision"]:
        tf.keras.mixed_precision.set_global_policy("mixed_float16")

    out = os.path.join(cfg["out_root"], cfg["backbone"])
    os.makedirs(out, exist_ok=True)

    paths, labels = list_samples(cfg["train_dir"])
    if cfg["smoke"]:                       # ponytail: uji end-to-end tanpa nunggu berjam-jam
        idx = np.random.RandomState(0).permutation(len(paths))[:300]
        paths, labels = paths[idx], labels[idx]
        cfg["epochs"] = 1

    if cfg["cv_folds"]:
        from sklearn.model_selection import StratifiedKFold
        skf = StratifiedKFold(n_splits=cfg["cv_folds"], shuffle=True, random_state=cfg["seed"])
        tr_i, va_i = list(skf.split(paths, labels))[cfg["fold"]]
        tr_p, va_p, tr_y, va_y = paths[tr_i], paths[va_i], labels[tr_i], labels[va_i]
        out = os.path.join(out, f"fold{cfg['fold']}")       # artefak per fold di subfolder
        os.makedirs(out, exist_ok=True)
        print(f"fold {cfg['fold']+1}/{cfg['cv_folds']}", flush=True)
    else:
        tr_p, va_p, tr_y, va_y = train_test_split(
            paths, labels, test_size=cfg["val_split"],
            stratify=labels, random_state=cfg["seed"])      # stratified (wajib guideline)

    # catat file val fold ini: sumber kebenaran utk OOF nanti (tak bergantung rekonstruksi split)
    with open(os.path.join(out, "val_files.txt"), "w") as f:
        f.write("\n".join(va_p))

    tr = make_ds(tr_p, tr_y, cfg["img_size"], cfg["batch_size"], True,  cfg["augment"], cfg["seed"],
                 erase_p=cfg["random_erase"])
    va = make_ds(va_p, va_y, cfg["img_size"], cfg["batch_size"], False, False, cfg["seed"])
    steps = -(-len(tr_p) // cfg["batch_size"])              # ceil
    print(f"train={len(tr_p)} val={len(va_p)} | steps/epoch={steps} batch={cfg['batch_size']} "
          f"img={cfg['img_size']} | epoch pertama lambat (tracing + baca disk), sabar", flush=True)

    cw = None
    if cfg["use_class_weight"]:
        w = compute_class_weight("balanced", classes=np.arange(len(CLASS_NAMES)), y=tr_y)
        cw = {i: float(w[i]) for i in range(len(w))}

    model = build_model(cfg)
    ls = cfg["label_smoothing"]
    if ls:
        # SparseCategoricalCrossentropy tidak punya label_smoothing -> one-hot dulu
        def loss(y, p):
            return tf.keras.losses.categorical_crossentropy(
                tf.one_hot(tf.cast(y, tf.int32), len(CLASS_NAMES)), p, label_smoothing=ls)
    else:
        loss = "sparse_categorical_crossentropy"

    if cfg["warmup_epochs"] and cfg["finetune"] != "freeze":
        # latih head dulu dgn backbone beku: head random tidak merusak bobot pretrained
        base = getattr(model, "backbone", None)          # keras_hub (ViT) punya .backbone
        if base is None:
            base = next(l for l in model.layers if isinstance(l, tf.keras.Model))  # keras.applications
        base.trainable = False
        model.compile(optimizer=tf.keras.optimizers.AdamW(1e-3, weight_decay=cfg["weight_decay"]),
                      loss=loss, metrics=["accuracy"])
        print(f"warmup {cfg['warmup_epochs']} epoch (head only, lr 1e-3)...", flush=True)
        model.fit(tr, validation_data=va, epochs=cfg["warmup_epochs"],
                  class_weight=cw, verbose=cfg["verbose"])
        _apply_finetune(base, cfg["finetune"])          # unfreeze sesuai mode, lanjut training utama

    model.compile(optimizer=make_optimizer(cfg), loss=loss, metrics=["accuracy"])

    ckpt = os.path.join(out, "_best.weights.h5")
    f1cb = MacroF1(va, ckpt, cfg["es_patience"])
    callbacks = [BatchLog(cfg["log_every"], steps), f1cb,
                 # history.csv ditulis live tiap epoch (crash-proof: kalau Colab mati, kurva tetap ada).
                 # f1cb di depan CSVLogger -> val_macro_f1 sudah masuk logs saat ditulis.
                 tf.keras.callbacks.CSVLogger(os.path.join(out, "history.csv"), append=False),
                 tf.keras.callbacks.ReduceLROnPlateau(
                     monitor="val_loss", factor=0.5, patience=cfg["rlrop_patience"], verbose=1)]

    t0 = time.time()
    hist = model.fit(tr, validation_data=va, epochs=cfg["epochs"],
                     class_weight=cw, callbacks=callbacks, verbose=cfg["verbose"])
    train_time = time.time() - t0

    model.load_weights(ckpt)                                # restore best (by macro-F1)

    # ---- evaluasi final di val ----
    y_true = np.concatenate([y.numpy() for _, y in va])
    t1 = time.time()
    y_prob = model.predict(va, verbose=0)
    infer_time = (time.time() - t1) / len(y_true) * 1000    # ms/gambar
    y_pred = y_prob.argmax(1)

    macro_f1 = f1_score(y_true, y_pred, average="macro")
    prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_true, y_pred, average="macro", zero_division=0)
    acc = float((y_pred == y_true).mean())

    _save_artifacts(cfg, out, model, hist, f1cb, y_true, y_pred,
                    dict(macro_f1=macro_f1, precision=prec, recall=rec, accuracy=acc,
                         val_loss=float(min(hist.history["val_loss"])),
                         train_acc=float(hist.history["accuracy"][f1cb.best_epoch]),
                         train_time_s=round(train_time, 1),
                         infer_ms_per_img=round(infer_time, 3),
                         params=int(model.count_params()),
                         best_epoch=f1cb.best_epoch + 1))

    if cfg["make_submission"]:
        _write_submission(cfg, model, out)

    print(f"\n=== {cfg['backbone']} | Macro F1={macro_f1:.4f} acc={acc:.4f} "
          f"best_epoch={f1cb.best_epoch+1} ===")
    return {"macro_f1": macro_f1, "accuracy": acc, "out": out}


def _save_artifacts(cfg, out, model, hist, f1cb, y_true, y_pred, metrics):
    import pandas as pd, matplotlib
    matplotlib.use("Agg"); import matplotlib.pyplot as plt

    model.save(os.path.join(out, f"{cfg['backbone']}.keras"))

    h = dict(hist.history)
    h["val_macro_f1"] = h.get("val_macro_f1", [])
    pd.DataFrame(h).to_csv(os.path.join(out, "history.csv"), index=False)

    rep = classification_report(y_true, y_pred, target_names=CLASS_NAMES,
                                digits=4, zero_division=0)
    with open(os.path.join(out, "classification_report.txt"), "w") as f:
        f.write(rep)

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    plt.imshow(cm, cmap="Blues")
    plt.xticks(range(3), CLASS_NAMES, rotation=45); plt.yticks(range(3), CLASS_NAMES)
    for i in range(3):
        for j in range(3):
            plt.text(j, i, cm[i, j], ha="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black")
    plt.ylabel("True"); plt.xlabel("Pred"); plt.title(cfg["backbone"]); plt.colorbar()
    plt.tight_layout(); plt.savefig(os.path.join(out, "confusion_matrix.png"), dpi=120); plt.close()

    ep = range(1, len(hist.history["accuracy"]) + 1)
    plt.figure(); plt.plot(ep, hist.history["accuracy"], label="train")
    plt.plot(ep, hist.history["val_accuracy"], label="val")
    plt.xlabel("epoch"); plt.ylabel("accuracy"); plt.legend(); plt.title("Accuracy")
    plt.savefig(os.path.join(out, "training_curve.png"), dpi=120); plt.close()

    plt.figure(); plt.plot(ep, hist.history["loss"], label="train")
    plt.plot(ep, hist.history["val_loss"], label="val")
    plt.xlabel("epoch"); plt.ylabel("loss"); plt.legend(); plt.title("Loss")
    plt.savefig(os.path.join(out, "loss_curve.png"), dpi=120); plt.close()

    params = {k: cfg[k] for k in ("backbone", "img_size", "batch_size", "epochs", "lr",
                                  "optimizer", "weight_decay", "label_smoothing", "dropout",
                                  "finetune", "use_class_weight", "augment", "seed",
                                  "warmup_epochs", "random_erase", "cv_folds", "fold")}
    params["pipeline_version"] = __version__
    params.update(metrics)
    with open(os.path.join(out, "parameter.json"), "w") as f:
        json.dump(params, f, indent=2)

    with open(os.path.join(out, "README.md"), "w") as f:
        f.write(f"""# {cfg['backbone']}

- Backbone: {cfg['backbone']} (ImageNet pretrained)
- Jumlah Parameter: {metrics['params']:,}
- Hardware: (isi manual: Colab T4 / RTX 3050)
- Training Time: {metrics['train_time_s']} s
- Best Epoch: {metrics['best_epoch']}
- **Macro F1: {metrics['macro_f1']:.4f}**
- Accuracy: {metrics['accuracy']:.4f}
- Inference: {metrics['infer_ms_per_img']} ms/img

## Kelebihan
-

## Kekurangan
-

## Catatan
- Config: img={cfg['img_size']} bs={cfg['batch_size']} lr={cfg['lr']} opt={cfg['optimizer']} finetune={cfg['finetune']}
""")

    if not os.path.exists(os.path.join(out, "notes.md")):
        with open(os.path.join(out, "notes.md"), "w") as f:
            f.write("# Notes\n\n## Dicoba\n\n## Gagal\n\n## Improvement\n\n## Masalah\n\n## Resource\n")

    os.remove(os.path.join(out, "_best.weights.h5"))


def _write_submission(cfg, model, out):
    """Prediksi folder test urut 1..N -> submission.csv (id,predicted)."""
    import pandas as pd
    files = sorted(os.listdir(cfg["test_dir"]), key=lambda x: int(x.split(".")[0]))
    paths = [os.path.join(cfg["test_dir"], f) for f in files]
    size = cfg["img_size"]

    def load(p):
        img = tf.io.decode_image(tf.io.read_file(p), channels=3, expand_animations=False)
        img = tf.image.resize(img, [size, size])
        return tf.ensure_shape(img, [size, size, 3])

    ds = tf.data.Dataset.from_tensor_slices(paths).map(
        load, num_parallel_calls=tf.data.AUTOTUNE).batch(cfg["batch_size"]).prefetch(tf.data.AUTOTUNE)
    pred = model.predict(ds, verbose=1).argmax(1)
    ids = [int(f.split(".")[0]) for f in files]
    pd.DataFrame({"id": ids, "predicted": pred}).to_csv(
        os.path.join(out, "submission.csv"), index=False)
    print(f"submission -> {os.path.join(out, 'submission.csv')}")


def run_cv(cfg=None):
    """5-fold (atau n-fold) stratified CV. Resume-aware: fold yang sudah punya
    parameter.json di-skip — aman kalau Colab mati di tengah (tinggal re-run)."""
    cfg = {**CONFIG, **(cfg or {})}
    folds = cfg["cv_folds"] or 5
    cfg["cv_folds"] = folds
    scores = []
    for i in range(folds):
        pj = os.path.join(cfg["out_root"], cfg["backbone"], f"fold{i}", "parameter.json")
        if os.path.exists(pj):
            with open(pj) as f:
                scores.append(json.load(f)["macro_f1"])
            print(f"fold {i+1}/{folds}: sudah selesai (macro_f1={scores[-1]:.4f}), skip", flush=True)
            continue
        scores.append(run({**cfg, "fold": i})["macro_f1"])
    arr = np.array(scores)
    summary = {"folds": folds,
               "macro_f1_per_fold": [round(float(s), 6) for s in scores],
               "macro_f1_mean": round(float(arr.mean()), 6),
               "macro_f1_std": round(float(arr.std()), 6)}
    with open(os.path.join(cfg["out_root"], cfg["backbone"], "cv_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n=== CV {folds}-fold Macro F1: {arr.mean():.4f} ± {arr.std():.4f} ===")
    return summary


def oof_predictions(cfg=None, tta=True):
    """Prediksi out-of-fold: tiap gambar train dinilai model fold yang TIDAK melihatnya saat training.
    Return (probs[N,3], labels[N]) — bahan LEGAL utk tuning threshold (bukan data test)."""
    from sklearn.model_selection import StratifiedKFold
    cfg = {**CONFIG, **(cfg or {})}
    paths, labels = list_samples(cfg["train_dir"])
    folds = cfg["cv_folds"] or 5
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=cfg["seed"])
    probs = np.zeros((len(paths), len(CLASS_NAMES)))
    pos = {p: i for i, p in enumerate(paths)}
    for i, (_, va_i) in enumerate(skf.split(paths, labels)):
        fold_dir = os.path.join(cfg["out_root"], cfg["backbone"], f"fold{i}")
        vf = os.path.join(fold_dir, "val_files.txt")
        if os.path.exists(vf):
            # sumber kebenaran: daftar file val yang DICATAT saat training (anti-leakage)
            with open(vf) as f:
                va_i = np.array([pos[p] for p in f.read().splitlines() if p])
        else:
            raise RuntimeError(
                f"{vf} tidak ada. Model fold ini dilatih sebelum pipeline v15 -> split tidak bisa "
                "direkonstruksi lintas sesi (urutan listdir berubah). OOF di model lama = LEAKAGE. "
                "Retrain dengan v15 dulu.")
        mp = os.path.join(fold_dir, f"{cfg['backbone']}.keras")
        print(f"fold {i}: {len(va_i)} gambar | {mp}", flush=True)
        m = tf.keras.models.load_model(mp, compile=False)
        ds = make_ds(paths[va_i], labels[va_i], cfg["img_size"], cfg["batch_size"],
                     False, False, cfg["seed"])                  # tanpa shuffle: urutan = va_i
        p = m.predict(ds, verbose=1)
        if tta:
            p = p + m.predict(ds.map(lambda x, y: (tf.image.flip_left_right(x), y)), verbose=1)
        probs[va_i] = p
        del m
    return probs, labels


def calibration_report(probs, y, bins=10):
    """Diagnosis kalibrasi di prediksi OOF: Log Loss, Brier, ECE, + tabel reliability.
    Model terkalibrasi baik = saat bilang 90% yakin, dia benar ~90% dari waktu."""
    from sklearn.metrics import log_loss
    p = probs / probs.sum(1, keepdims=True)          # normalisasi (OOF = jumlah 2 view TTA)
    onehot = np.eye(len(CLASS_NAMES))[y]
    ll = log_loss(y, p)
    brier = float(np.mean(np.sum((p - onehot) ** 2, axis=1)))   # multiclass Brier

    conf, pred = p.max(1), p.argmax(1)
    correct = (pred == y).astype(float)
    ece, rows = 0.0, []
    for lo in np.linspace(0, 1, bins + 1)[:-1]:
        m = (conf >= lo) & (conf < lo + 1 / bins + (lo + 1 / bins >= 1) * 1e-9)
        if m.sum() == 0:
            continue
        acc, avg_conf = correct[m].mean(), conf[m].mean()
        ece += m.mean() * abs(acc - avg_conf)
        rows.append(f"  conf {lo:.1f}-{lo + 1/bins:.1f}: n={m.sum():5d}  akurasi={acc:.3f}  "
                    f"rata2 conf={avg_conf:.3f}  gap={acc - avg_conf:+.3f}")
    print(f"Log Loss={ll:.4f} | Brier={brier:.4f} | ECE={ece:.4f}")
    print("Reliability (gap>0 = under-confident, gap<0 = over-confident):")
    print("\n".join(rows))
    return {"log_loss": ll, "brier": brier, "ece": ece}


def tune_class_weights(probs, y, grid=None):
    """Grid search bobot per kelas: prediksi = argmax(w * p). Menggeser ambang keputusan
    (mis. w_Organic < 1 -> lebih pelit memilih Organic). Dituning di OOF, dipakai di test."""
    grid = grid if grid is not None else np.arange(0.70, 1.32, 0.05)
    base = f1_score(y, probs.argmax(1), average="macro")
    best_f1, best_w = base, np.ones(3)
    for wr in grid:                    # w_Electronic dikunci 1 (hanya rasio yang berpengaruh)
        for wo in grid:
            w = np.array([wr, 1.0, wo])
            f1 = f1_score(y, (probs * w).argmax(1), average="macro")
            if f1 > best_f1:
                best_f1, best_w = f1, w
    print(f"OOF macro F1: baseline={base:.4f} -> tuned={best_f1:.4f} "
          f"(w_R={best_w[0]:.2f} w_E=1.00 w_O={best_w[2]:.2f})")
    return best_w


def submission_ensemble(model_paths, cfg=None, out_csv="submission.csv", tta=True, class_w=None):
    """Ensemble model-model ber-resolusi sama (pakai cfg['img_size'])."""
    cfg = {**CONFIG, **(cfg or {})}
    return submission_ensemble_multi([(mp, cfg["img_size"]) for mp in model_paths],
                                     cfg, out_csv=out_csv, tta=tta, class_w=class_w)


def submission_ensemble_multi(entries, cfg=None, out_csv="submission.csv", tta=True, class_w=None):
    """Ensemble lintas arsitektur/resolusi. entries = list (path_model, img_size).
    Tiap model menerima gambar di resolusi training-nya sendiri; probabilitas dirata-rata."""
    import pandas as pd
    cfg = {**CONFIG, **(cfg or {})}
    files = sorted(os.listdir(cfg["test_dir"]), key=lambda x: int(x.split(".")[0]))
    fpaths = [os.path.join(cfg["test_dir"], f) for f in files]

    def make(size):
        def load(p):
            img = tf.io.decode_image(tf.io.read_file(p), channels=3, expand_animations=False)
            img = tf.image.resize(img, [size, size])
            return tf.ensure_shape(img, [size, size, 3])
        return tf.data.Dataset.from_tensor_slices(fpaths).map(
            load, num_parallel_calls=tf.data.AUTOTUNE).batch(cfg["batch_size"]).prefetch(tf.data.AUTOTUNE)

    probs = 0.0
    for size in sorted({s for _, s in entries}):        # satu dataset per resolusi (hemat decode)
        ds = make(size)
        ds_flip = ds.map(lambda x: tf.image.flip_left_right(x))
        for mp, s in entries:
            if s != size:
                continue
            print(f"model: {mp} @ {size}px", flush=True)
            m = tf.keras.models.load_model(mp, compile=False)  # tanpa loss/optimizer: cukup predict
            probs = probs + m.predict(ds, verbose=1)
            if tta:
                probs = probs + m.predict(ds_flip, verbose=1)
            del m
    if class_w is not None:
        probs = probs * np.asarray(class_w)    # threshold shift hasil tune_class_weights (OOF)
    pred = probs.argmax(1)
    ids = [int(f.split(".")[0]) for f in files]
    pd.DataFrame({"id": ids, "predicted": pred}).to_csv(out_csv, index=False)

    # file pendamping berisi confidence (JANGAN disubmit - format panitia hanya id,predicted)
    pn = probs / probs.sum(1, keepdims=True)
    conf_csv = out_csv.replace(".csv", "_conf.csv")
    pd.DataFrame({"id": ids, "predicted": pred, "conf": pn.max(1).round(4),
                  "p_recyclable": pn[:, 0].round(4), "p_electronic": pn[:, 1].round(4),
                  "p_organic": pn[:, 2].round(4)}).to_csv(conf_csv, index=False)

    print(f"-> {out_csv} | distribusi prediksi (R/E/O): {np.bincount(pred, minlength=3)}")
    print(f"-> {conf_csv} | conf rata2={pn.max(1).mean():.4f} | "
          f"ragu (conf<0.7): {(pn.max(1) < 0.7).sum()} gambar")
    return pred


if __name__ == "__main__":
    # smoke test: end-to-end di 300 gambar, 1 epoch — cek pipeline nggak error & artefak lengkap
    r = run({"smoke": True, "epochs": 1})
    assert os.path.exists(os.path.join(r["out"], "EfficientNetB0.keras"))
    assert 0.0 <= r["macro_f1"] <= 1.0
    print("smoke OK")
