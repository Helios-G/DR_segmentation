"""Find samples the model PREDICTS as class 4 (most severe DR / PDR) and
visualise what regions drive that decision."""

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
OUT_DIR = os.path.join(PROJECT, "gradcam_out", "class4")
INPUT_SIZE = 256
TARGET_CLASS = 4

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


def scan_for_class(model, labels, target_cls, n_needed, max_scan, seed):
    """Iterate through images, batch-predict, collect those with argmax == target."""
    rng = np.random.default_rng(seed)
    # bias toward true-severe samples first to improve hit-rate
    high = [n for n, l in labels.items() if l >= 3]
    low = [n for n, l in labels.items() if l < 3]
    rng.shuffle(high)
    rng.shuffle(low)
    order = high + low

    found = []
    batch_imgs, batch_meta = [], []
    BATCH = 32
    scanned = 0
    for name in order:
        if scanned >= max_scan or len(found) >= n_needed:
            break
        path = os.path.join(TRAIN_DIR, name + ".jpeg")
        rgb, x = load_image(path)
        if x is None:
            continue
        batch_imgs.append(x)
        batch_meta.append((name, labels[name], rgb, x))
        scanned += 1
        if len(batch_imgs) == BATCH or len(found) >= n_needed:
            preds = model.predict(np.stack(batch_imgs), verbose=0)
            for (n, lvl, rgb_img, xp), prob in zip(batch_meta, preds):
                if int(prob.argmax()) == target_cls and len(found) < n_needed:
                    found.append((n, lvl, rgb_img, xp, prob))
                    print(f"  hit: {n} true={lvl} probs={np.round(prob,3).tolist()}")
            batch_imgs, batch_meta = [], []
    print(f"Scanned {scanned}, found {len(found)} pred=={target_cls}")
    return found


def gradcam(grad_model, x, class_idx, plus_plus=False):
    x_t = tf.convert_to_tensor(x[None, ...])
    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(x_t, training=False)
        score = preds[:, class_idx]
    grads = tape.gradient(score, conv_out)  # (1,H,W,C)
    if plus_plus:
        # Grad-CAM++ weights
        grads2 = grads ** 2
        grads3 = grads ** 3
        sum_a = tf.reduce_sum(conv_out, axis=(1, 2), keepdims=True)
        denom = 2.0 * grads2 + sum_a * grads3
        denom = tf.where(denom != 0.0, denom, tf.ones_like(denom))
        alpha = grads2 / denom
        weights = tf.reduce_sum(alpha * tf.nn.relu(grads), axis=(1, 2))  # (1,C)
    else:
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


def contour_overlay(rgb, cam_up, levels=(0.4, 0.6, 0.8)):
    fig_rgb = rgb.copy()
    for lv, color in zip(levels, [(255, 255, 0), (255, 128, 0), (255, 0, 0)]):
        m = (cam_up >= lv).astype(np.uint8)
        contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(fig_rgb, contours, -1, color, 2)
    return fig_rgb


def build_grad_model_with_inner(model, inner_layer_name):
    """Expose an InceptionV3-internal layer as an extra output of the full model
    by rebuilding the outer graph and rewiring all post-inception layers."""
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


def get_learned_attention(model, x):
    sub = tf.keras.Model(model.input, model.get_layer("conv2d_1087").output)
    att = sub(x[None, ...], training=False)[0, ..., 0].numpy()
    if att.max() > 0:
        att = att / att.max()
    return att


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8, help="number of class-4 hits to visualise")
    ap.add_argument("--max-scan", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    print("Loading model on GPU...")
    model = load_dr_model()

    grad_mid = build_grad_model_with_inner(model, "mixed7")
    grad_deep = tf.keras.Model(
        model.input,
        [model.get_layer("multiply_10").output, model.output],
    )

    labels = load_labels()
    print(f"Scanning train set for predicted class {TARGET_CLASS}...")
    hits = scan_for_class(model, labels, TARGET_CLASS, args.n, args.max_scan, args.seed)
    if not hits:
        print("No class-4 predictions found — try increasing --max-scan")
        return

    # Per-image detail figure
    for img_name, true_lvl, rgb, x, prob in hits:
        cam_mid = upscale(gradcam(grad_mid, x, TARGET_CLASS, plus_plus=False), INPUT_SIZE)
        cam_mid_pp = upscale(gradcam(grad_mid, x, TARGET_CLASS, plus_plus=True), INPUT_SIZE)
        cam_deep = upscale(gradcam(grad_deep, x, TARGET_CLASS, plus_plus=False), INPUT_SIZE)
        att = upscale(get_learned_attention(model, x), INPUT_SIZE)

        fig, axes = plt.subplots(1, 6, figsize=(22, 4.2))
        axes[0].imshow(rgb)
        axes[0].set_title(
            f"{img_name}\ntrue={true_lvl} ({CLASS_NAMES[true_lvl]})\n"
            f"pred=4 p={prob[4]:.2f}",
            fontsize=10,
        )
        axes[1].imshow(overlay_heat(rgb, cam_deep))
        axes[1].set_title("Grad-CAM @ multiply_10\n(6x6, last attention feat.)", fontsize=10)
        axes[2].imshow(overlay_heat(rgb, cam_mid))
        axes[2].set_title("Grad-CAM @ mixed7\n(14x14, mid-level)", fontsize=10)
        axes[3].imshow(overlay_heat(rgb, cam_mid_pp))
        axes[3].set_title("Grad-CAM++ @ mixed7\n(sharper localisation)", fontsize=10)
        axes[4].imshow(att, cmap="hot")
        axes[4].set_title("Learned attention map\n(conv2d_1087, sigmoid)", fontsize=10)
        axes[5].imshow(contour_overlay(rgb, cam_mid_pp))
        axes[5].set_title("Contours: 0.4 / 0.6 / 0.8\n(on Grad-CAM++)", fontsize=10)
        for a in axes:
            a.axis("off")

        out_path = os.path.join(OUT_DIR, f"pred4_{img_name}.png")
        plt.tight_layout()
        plt.savefig(out_path, dpi=130, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved {out_path}")

    # also save a thumbnail index
    thumbs = []
    for img_name, true_lvl, rgb, x, prob in hits:
        cam_mid_pp = upscale(gradcam(grad_mid, x, TARGET_CLASS, plus_plus=True), INPUT_SIZE)
        thumbs.append((img_name, true_lvl, rgb, cam_mid_pp, prob))
    fig, axes = plt.subplots(len(thumbs), 2, figsize=(7, 3 * len(thumbs)))
    if len(thumbs) == 1:
        axes = axes[None, :]
    for i, (n, t, rgb, c, p) in enumerate(thumbs):
        axes[i, 0].imshow(rgb)
        axes[i, 0].set_title(f"{n} true={t} pred=4 p={p[4]:.2f}", fontsize=9)
        axes[i, 1].imshow(overlay_heat(rgb, c))
        axes[i, 1].set_title("Grad-CAM++ @ mixed7", fontsize=9)
        for j in (0, 1):
            axes[i, j].axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "_index.png"), dpi=130, bbox_inches="tight")
    print(f"\nIndex saved: {os.path.join(OUT_DIR, '_index.png')}")


if __name__ == "__main__":
    main()
