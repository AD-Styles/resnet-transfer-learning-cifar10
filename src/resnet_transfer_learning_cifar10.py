# ==========================================
# [Cell 1] Setup, Hyperparameters, and Data
# ==========================================

import os
import time
import platform

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50, ResNet101
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split

SEED        = 42
IMG_SIZE    = 224
BATCH_SIZE  = 32
EPOCHS_A    = 5
EPOCHS_B    = 5
UNFREEZE_N  = 30
LR_A        = 1e-3
LR_B        = 5e-6
CLIP_NORM   = 1.0

OUTPUT_DIR  = "./results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

CLASS_NAMES = ["airplane", "automobile", "bird", "cat", "deer",
               "dog", "frog", "horse", "ship", "truck"]

# System Config
print("=" * 60)
print("SYSTEM CONFIGURATION")
print("=" * 60)
print(f"OS            : {platform.system()} {platform.release()}")
print(f"TensorFlow    : {tf.__version__}")
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    print(f"GPU detected  : {len(gpus)}")
    for g in gpus:
        tf.config.experimental.set_memory_growth(g, True)
else:
    print("GPU detected  : None — running on CPU")

# Data Loading
(x_train_full, y_train_full), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
y_train_full = y_train_full.squeeze().astype("int64")
y_test       = y_test.squeeze().astype("int64")

x_train, x_val, y_train, y_val = train_test_split(
    x_train_full, y_train_full, test_size=0.1, random_state=SEED, stratify=y_train_full
)

AUTOTUNE = tf.data.AUTOTUNE

def preprocess(x, y):
    x = tf.cast(x, tf.float32)
    x = tf.image.resize(x, (IMG_SIZE, IMG_SIZE), method="bilinear")
    return x, y

def make_dataset(x, y, shuffle=False):
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    if shuffle:
        ds = ds.shuffle(10_000, seed=SEED, reshuffle_each_iteration=True)
    return ds.map(preprocess, num_parallel_calls=AUTOTUNE).batch(BATCH_SIZE).prefetch(AUTOTUNE)

train_ds = make_dataset(x_train, y_train, shuffle=True)
val_ds   = make_dataset(x_val, y_val)
test_ds  = make_dataset(x_test, y_test)

print("\nData pipeline ready.")


# ==========================================
# [Cell 2] Model Builder & Training Helpers
# ==========================================

def build_model(backbone_fn, num_classes=10, img_size=IMG_SIZE):
    inputs = layers.Input(shape=(img_size, img_size, 3))
    x = layers.Lambda(tf.keras.applications.resnet.preprocess_input, name="resnet_preprocess")(inputs)
    backbone = backbone_fn(include_top=False, weights="imagenet", input_tensor=x)
    backbone.trainable = False

    x = backbone.output
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dense(256, activation="relu", name="head_dense")(x)
    x = layers.Dropout(0.5, name="head_dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = models.Model(inputs=backbone.input, outputs=outputs)
    return model, backbone

def get_callbacks():
    return [
        EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=1, min_lr=1e-6, verbose=1)
    ]

def run_stage_a(model, backbone, label):
    print(f"\n[{label}] Stage A: Head training (backbone frozen)")
    model.compile(optimizer=tf.keras.optimizers.Adam(LR_A), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_A, callbacks=get_callbacks(), verbose=1)
    
    metrics_raw = dict(zip(model.metrics_names, model.evaluate(test_ds, verbose=0)))
    metrics = {"loss": metrics_raw["loss"]}
    for k, v in metrics_raw.items():
        if "acc" in k.lower(): metrics["accuracy"] = v; break

    with tf.keras.utils.custom_object_scope({'preprocess_input': tf.keras.applications.resnet.preprocess_input}):
        snapshot = models.clone_model(model)
        snapshot.build((None, IMG_SIZE, IMG_SIZE, 3))
        snapshot.set_weights(model.get_weights())

    return history, metrics, snapshot

def run_stage_b(model, backbone, label):
    print(f"\n[{label}] Stage B: Fine-tuning (top {UNFREEZE_N} unfrozen)")
    for layer in backbone.layers[:-UNFREEZE_N]: layer.trainable = False
    for layer in backbone.layers[-UNFREEZE_N:]: layer.trainable = True

    model.compile(optimizer=tf.keras.optimizers.Adam(LR_B, clipnorm=CLIP_NORM), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_B, callbacks=get_callbacks(), verbose=1)

    metrics_raw = dict(zip(model.metrics_names, model.evaluate(test_ds, verbose=0)))
    metrics = {"loss": metrics_raw["loss"]}
    for k, v in metrics_raw.items():
        if "acc" in k.lower(): metrics["accuracy"] = v; break

    return history, metrics

print("Models and helpers initialized.")


# ==========================================
# [Cell 3] Execute Training
# ==========================================

# Build ResNet50
resnet50_model, resnet50_base = build_model(ResNet50)
print(f"ResNet50 Total params: {resnet50_model.count_params() / 1e6:.2f} M")

# Build ResNet101
resnet101_model, resnet101_base = build_model(ResNet101)
print(f"ResNet101 Total params: {resnet101_model.count_params() / 1e6:.2f} M")

# Train ResNet50
hist50_a, test50_a, snap50_a = run_stage_a(resnet50_model, resnet50_base,  "ResNet50")
hist50_b, test50_b           = run_stage_b(resnet50_model, resnet50_base,  "ResNet50")

# Train ResNet101
hist101_a, test101_a, snap101_a = run_stage_a(resnet101_model, resnet101_base, "ResNet101")
hist101_b, test101_b            = run_stage_b(resnet101_model, resnet101_base, "ResNet101")

print("\nTraining completed for both models.")


# ==========================================
# [Cell 4] Evaluation & Visualization (KeyError 및 OOM 완전 방지 패치)
# ==========================================

import gc

print("Keras 3 호환성 패치 적용: 메모리 내 모델들로부터 안전하게 지표를 복구합니다...")

def safe_evaluate(model):
    """
    Keras 3 환경에서 정확도(accuracy) 누락 없이 확실하게 평가 지표를 추출하는 안전 헬퍼 함수.
    snapshot 모델은 compile이 안 되어 있을 수 있으므로 먼저 compile 처리.
    """
    if not hasattr(model, 'optimizer') or model.optimizer is None:
        model.compile(loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        
    # return_dict=True를 사용하면 강제로 올바른 메트릭 이름의 딕셔너리를 반환함
    res = model.evaluate(test_ds, verbose=0, return_dict=True)
    
    metrics = {"loss": res.get("loss", 0.0)}
    for k, v in res.items():
        if "acc" in k.lower():
            metrics["accuracy"] = v
            break
            
    # 만약 여전히 accuracy를 찾지 못했다면 강제로 두 번째 지표(보통 정확도)를 할당
    if "accuracy" not in metrics:
        metrics["accuracy"] = list(res.values())[1] if len(res) > 1 else 0.0
        
    return metrics

# 1. 램(RAM)에 살아있는 모델들로 평가 지표 재계산
test50_a  = safe_evaluate(snap50_a)
test50_b  = safe_evaluate(resnet50_model)
test101_a = safe_evaluate(snap101_a)
test101_b = safe_evaluate(resnet101_model)

print(f"✅ 지표 복구 완료: ResNet50_StageB Acc = {test50_b['accuracy']:.4f}")

# 2. 시각화 및 벤치마크
PALETTE = {"stage_a_train": "#3498db", "stage_a_val": "#85c1e9", "stage_b_train": "#e67e22", "stage_b_val": "#f0b27a"}

def _resolve_acc_key(history_dict):
    for candidate in ("accuracy", "sparse_categorical_accuracy"):
        if candidate in history_dict: return candidate
    for k in history_dict:
        if "acc" in k.lower() and not k.startswith("val_"): return k
    return "accuracy"

def plot_learning_curves(hist_a, hist_b, model_name):
    acc_key = _resolve_acc_key(hist_a.history)
    def stitch(key): return hist_a.history[key] + hist_b.history[key]
    
    epochs_a = len(hist_a.history["loss"])
    total_ep = epochs_a + len(hist_b.history["loss"])
    x_axis = range(1, total_ep + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"{model_name} — Learning Curves", fontsize=13, fontweight="bold")

    for ax, metric, ylabel in zip(axes, ["loss", acc_key], ["Loss", "Accuracy"]):
        ax.plot(x_axis[:epochs_a], stitch(metric)[:epochs_a], color=PALETTE["stage_a_train"], marker="o", label="Train (A)")
        ax.plot(x_axis[:epochs_a], stitch(f"val_{metric}")[:epochs_a], color=PALETTE["stage_a_val"], marker="o", linestyle="--", label="Val (A)")
        ax.plot(x_axis[epochs_a:], stitch(metric)[epochs_a:], color=PALETTE["stage_b_train"], marker="s", label="Train (B)")
        ax.plot(x_axis[epochs_a:], stitch(f"val_{metric}")[epochs_a:], color=PALETTE["stage_b_val"], marker="s", linestyle="--", label="Val (B)")
        ax.axvline(x=epochs_a + 0.5, color="gray", linestyle=":", label="Stage A→B")
        ax.set_xlabel("Epoch"); ax.set_ylabel(ylabel); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, f"fig_02_curves_{model_name}.png")
    plt.savefig(path, dpi=120, bbox_inches="tight"); plt.close()

def plot_accuracy_comparison(t50_a, t50_b, t101_a, t101_b):
    labels = ["ResNet50\nStage A", "ResNet50\nStage B", "ResNet101\nStage A", "ResNet101\nStage B"]
    accs = [t50_a["accuracy"]*100, t50_b["accuracy"]*100, t101_a["accuracy"]*100, t101_b["accuracy"]*100]
    colors = ["#3498db", "#2980b9", "#2ecc71", "#27ae60"]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(labels, accs, color=colors, alpha=0.85, edgecolor="black", width=0.5)
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()+0.05, f"{acc:.2f}%", ha="center", va="bottom", fontweight="bold")
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_ylim([min(accs)-3, min(100, max(accs)+2)]); ax.grid(axis="y", alpha=0.3)
    
    path = os.path.join(OUTPUT_DIR, "fig_03_accuracy_comparison.png")
    plt.savefig(path, dpi=120, bbox_inches="tight"); plt.close()

def plot_efficiency_radar(p50, p101, t50, t101, ms50, ms101):
    metrics = ["Parameters (M)", "Test Accuracy (%)", "Speed (ms/img, lower=better)"]
    v50 = [p50, t50["accuracy"]*100, ms50]
    v101 = [p101, t101["accuracy"]*100, ms101]

    x = np.arange(len(metrics)); width = 0.35
    fig, ax = plt.subplots(figsize=(11, 5))
    b1 = ax.barh(x + width/2, v50, width, label="ResNet50", color="#3498db", edgecolor="black")
    b2 = ax.barh(x - width/2, v101, width, label="ResNet101", color="#2ecc71", edgecolor="black")

    for bars, vals in [(b1, v50), (b2, v101)]:
        for bar, v in zip(bars, vals):
            ax.text(v + max(vals)*0.01, bar.get_y() + bar.get_height()/2, f"{v:.2f}", va="center", fontsize=9)

    ax.set_yticks(x); ax.set_yticklabels(metrics); ax.legend(); ax.grid(axis="x", alpha=0.3)
    path = os.path.join(OUTPUT_DIR, "fig_04_efficiency_comparison.png")
    plt.savefig(path, dpi=120, bbox_inches="tight"); plt.close()

def benchmark(model, dataset, label, warmup=5, timed=40):
    it = iter(dataset)
    for _ in range(warmup):
        try: xb, _ = next(it)
        except StopIteration: it = iter(dataset); xb, _ = next(it)
        _ = float(np.sum(model(xb, training=False).numpy()))
        
    it = iter(dataset); total_imgs = 0
    t0 = time.perf_counter()
    for _ in range(timed):
        try: xb, _ = next(it)
        except StopIteration: it = iter(dataset); xb, _ = next(it)
        _ = float(np.sum(model(xb, training=False).numpy()))
        total_imgs += int(xb.shape[0])
    
    ms_per_img = (time.perf_counter() - t0) * 1000 / max(total_imgs, 1)
    print(f"[{label}] {ms_per_img:.3f} ms/image")
    return ms_per_img

gc.collect()

print("\nRunning benchmarks and generating plots...")
benchmark_ds = test_ds.unbatch().batch(16).prefetch(tf.data.AUTOTUNE)

resnet50_ms  = benchmark(resnet50_model, benchmark_ds, "ResNet50")
resnet101_ms = benchmark(resnet101_model, benchmark_ds, "ResNet101")

plot_learning_curves(hist50_a, hist50_b, "ResNet50")
plot_learning_curves(hist101_a, hist101_b, "ResNet101")
plot_accuracy_comparison(test50_a, test50_b, test101_a, test101_b)
plot_efficiency_radar(resnet50_model.count_params()/1e6, resnet101_model.count_params()/1e6, test50_b, test101_b, resnet50_ms, resnet101_ms)

print("\nDone. Output files saved to /kaggle/working/")
