"""For every class 0..4 collect a few high-confidence predictions and produce
Grad-CAM-based pseudo-segmentation visualisations."""

import gpu_bootstrap  # noqa: F401
import os
import csv
import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import preprocess_input
from model_loader import load_dr_model

PROJECT = r"C:\Users\hyobi\PycharmProjects\DR_segmentation"
TRAIN_DIR = os.path.join(PROJECT, "train")
LABELS_CSV = os.path.join(PROJECT, "trainLabels.csv")
OUT_DIR = os.path.join(PROJECT, "gradcam_out", "per_class")
INPUT_SIZE = 256

CLASS_NAMES = {
    0: "No DR",
    1: "Mild NPDR",
    2: "Moderate NPDR",
    3: "Severe NPDR",
    4: "Proliferative DR",
}


def load_labels():
    out = {}
    with open(LABELS_CSV, "r", newline="") as f:
        for row in csv.DictReader(f):
            out[row["image"]] = int(row["level"])
    return out


def load_image(path):
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        return None, None
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_AREA)
    x = preprocess_input(resized.astype(np.float32).copy())
    return resized, x


def scan_per_class(model, labels, per_class, max_scan, seed):
    """Collect `per_class` high-confidence predictions for every class 0..4.
    Prefer samples whose true label matches the predicted class first."""
    rng = np.random.default_rng(seed)

    # iterate roughly balanced over true labels so we hit every predicted class
    pools = {c: [] for c in range(5)}
    for n, l in labels.items():
        pools[l].append(n)
    for c in pools:
        rng.shuffle(pools[c])

    # interleave one from each class
    order = []
    max_len = max(len(v) for v in pools.values())
    for i in range(max_len):
        for c in range(5):
            if i < len(pools[c]):
                order.append(pools[c][i])

    by_pred = {c: [] for c in range(5)}  # list of (name, true, rgb, x, prob)
    batch_imgs, batch_meta = [], []
    BATCH = 32
    scanned = 0
    for name in order:
        if scanned >= max_scan:
            break
        if all(len(by_pred[c]) >= per_class for c in range(5)):
            break
        path = os.path.join(TRAIN_DIR, name + ".jpeg")
        rgb, x = load_image(path)
        if x is None:
            continue
        batch_imgs.append(x)
        batch_meta.append((name, labels[name], rgb, x))
        scanned += 1
        if len(batch_imgs) == BATCH:
            preds = model.predict(np.stack(batch_imgs), verbose=0)
            for (n, lvl, rgb_img, xp), prob in zip(batch_meta, preds):
                pc = int(prob.argmax())
                if len(by_pred[pc]) < per_class:
                    by_pred[pc].append((n, lvl, rgb_img, xp, prob))
            batch_imgs, batch_meta = [], []

    if batch_imgs:
        preds = model.predict(np.stack(batch_imgs), verbose=0)
        for (n, lvl, rgb_img, xp), prob in zip(batch_meta, preds):
            pc = int(prob.argmax())
            if len(by_pred[pc]) < per_class:
                by_pred[pc].append((n, lvl, rgb_img, xp, prob))

    # rank each class by confidence
    for c in range(5):
        by_pred[c].sort(key=lambda t: t[4][c], reverse=True)
        by_pred[c] = by_pred[c][:per_class]
        print(f"  pred={c} ({CLASS_NAMES[c]}): {len(by_pred[c])} samples")
    return by_pred


def build_grad_model_with_inner(model, inner_layer_name):
    inc = model.get_layer("inception_v3")
    inc_dual = tf.keras.Model(
        inc.input,
        [inc.get_layer(inner_layer_name).output, inc.output],
        name=f"inc_dual_{inner_layer_name}",
    )
    mid_out, inc_final = inc_dual(model.input)
    bn = model.get_layer("batch_normalization_1044")(inc_final)
    d0 = model.get_layer("dropout_30")(bn)
    c1 = model.get_layer("conv2d_1084")(d0)
    c2 = model.get_layer("conv2d_1085")(c1)
    c3 = model.get_layer("conv2d_1086")(c2)
    att = model.get_layer("conv2d_1087")(c3)
    scaled = model.get_layer("conv2d_1088")(att)
    mul = model.get_layer("multiply_10")([scaled, bn])
    gap1 = model.get_layer("global_average_pooling2d_20")(mul)
    gap2 = model.get_layer("global_average_pooling2d_21")(scaled)
    rs = model.get_layer("RescaleGAP")([gap1, gap2])
    y = model.get_layer("dropout_31")(rs)
    y = model.get_layer("dense_20")(y)
    y = model.get_layer("dropout_32")(y)
    y = model.get_layer("dense_21")(y)
    return tf.keras.Model(model.input, [mid_out, y], name=f"grad_{inner_layer_name}")


def gradcam(grad_model, x, class_idx):
    x_t = tf.convert_to_tensor(x[None, ...])
    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(x_t, training=False)
        score = preds[:, class_idx]
    grads = tape.gradient(score, conv_out)
    weights = tf.reduce_mean(grads, axis=(1, 2))
    cam = tf.reduce_sum(conv_out * weights[:, None, None, :], axis=-1)
    cam = tf.nn.relu(cam)[0].numpy()
    if cam.max() > 0:
        cam = cam / cam.max()
    return cam


def upscale(cam, size):
    return cv2.resize(cam, (size, size), interpolation=cv2.INTER_CUBIC)


def overlay_heat(rgb, cam_up, alpha=0.45):
    heat = cv2.applyColorMap(np.uint8(255 * np.clip(cam_up, 0, 1)), cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    return np.uint8((1 - alpha) * rgb + alpha * heat)


def make_mask(cam_up, threshold):
    return (cam_up >= threshold).astype(np.uint8) * 255


def mask_outline(rgb, mask):
    out = rgb.copy()
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, (0, 255, 0), 2)
    return out


def darken_outside(rgb, mask, factor=0.2):
    out = rgb.copy()
    out[mask == 0] = (out[mask == 0] * factor).astype(np.uint8)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=3)
    ap.add_argument("--max-scan", type=int, default=2500)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=2)
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    print("Loading model on GPU...")
    model = load_dr_model()
    grad_model = build_grad_model_with_inner(model, "mixed7")

    labels = load_labels()
    print(f"Scanning for predictions in each class (max-scan={args.max_scan})...")
    by_pred = scan_per_class(model, labels, args.per_class, args.max_scan, args.seed)

    # Per-class figure: rows = samples, cols = original | heatmap | mask | mask outline | masked
    for cls in range(5):
        hits = by_pred[cls]
        if not hits:
            print(f"  no hits for class {cls}, skipping")
            continue
        cols = 5
        rows = len(hits)
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.3, rows * 3.3))
        if rows == 1:
            axes = axes[None, :]
        fig.suptitle(
            f"Class {cls} — {CLASS_NAMES[cls]}  (Grad-CAM @ mixed7, threshold={args.threshold})",
            fontsize=14,
        )
        for i, (name, true_lvl, rgb, x, prob) in enumerate(hits):
            cam = upscale(gradcam(grad_model, x, cls), INPUT_SIZE)
            heat = overlay_heat(rgb, cam)
            mask = make_mask(cam, args.threshold)
            outline = mask_outline(rgb, mask)
            masked = darken_outside(rgb, mask)
            axes[i, 0].imshow(rgb)
            axes[i, 0].set_title(
                f"{name}\ntrue={true_lvl} pred={cls} p={prob[cls]:.2f}", fontsize=9
            )
            axes[i, 1].imshow(heat)
            axes[i, 1].set_title("Grad-CAM overlay", fontsize=9)
            axes[i, 2].imshow(mask, cmap="gray")
            axes[i, 2].set_title(f"pseudo-mask (t={args.threshold})", fontsize=9)
            axes[i, 3].imshow(outline)
            axes[i, 3].set_title("mask outline", fontsize=9)
            axes[i, 4].imshow(masked)
            axes[i, 4].set_title("segmented region", fontsize=9)
            for j in range(cols):
                axes[i, j].axis("off")
        out_path = os.path.join(OUT_DIR, f"class{cls}_{CLASS_NAMES[cls].replace(' ','_')}.png")
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        plt.savefig(out_path, dpi=130, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved {out_path}")

    # Combined overview: 1 sample per class side-by-side
    fig, axes = plt.subplots(5, 4, figsize=(13, 16))
    fig.suptitle("All classes — best representative (Grad-CAM @ mixed7)", fontsize=14)
    for cls in range(5):
        if not by_pred[cls]:
            for j in range(4):
                axes[cls, j].axis("off")
            continue
        name, true_lvl, rgb, x, prob = by_pred[cls][0]
        cam = upscale(gradcam(grad_model, x, cls), INPUT_SIZE)
        heat = overlay_heat(rgb, cam)
        mask = make_mask(cam, args.threshold)
        outline = mask_outline(rgb, mask)
        axes[cls, 0].imshow(rgb)
        axes[cls, 0].set_title(
            f"Class {cls}: {CLASS_NAMES[cls]}\n{name} true={true_lvl} p={prob[cls]:.2f}",
            fontsize=10,
        )
        axes[cls, 1].imshow(heat)
        axes[cls, 1].set_title("Grad-CAM", fontsize=10)
        axes[cls, 2].imshow(mask, cmap="gray")
        axes[cls, 2].set_title(f"mask (t={args.threshold})", fontsize=10)
        axes[cls, 3].imshow(outline)
        axes[cls, 3].set_title("outline", fontsize=10)
        for j in range(4):
            axes[cls, j].axis("off")
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    overview_path = os.path.join(OUT_DIR, "overview_all_classes.png")
    plt.savefig(overview_path, dpi=130, bbox_inches="tight")
    print(f"\nOverview saved: {overview_path}")


if __name__ == "__main__":
    main()
