import gpu_bootstrap  # noqa: F401
import os
import csv
import random
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
OUT_DIR = os.path.join(PROJECT, "gradcam_out")
INPUT_SIZE = 256
DEFAULT_TARGET_LAYER = "multiply_10"


def load_labels():
    table = {}
    with open(LABELS_CSV, "r", newline="") as f:
        for row in csv.DictReader(f):
            table[row["image"]] = int(row["level"])
    return table


def pick_samples(labels, per_class=2, seed=0):
    rng = random.Random(seed)
    by_cls = {c: [] for c in range(5)}
    for name, lvl in labels.items():
        if os.path.exists(os.path.join(TRAIN_DIR, name + ".jpeg")):
            by_cls[lvl].append(name)
    picks = []
    for c in range(5):
        rng.shuffle(by_cls[c])
        picks.extend((n, c) for n in by_cls[c][:per_class])
    return picks


def load_image(path):
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_AREA)
    arr = resized.astype(np.float32)
    x = preprocess_input(arr.copy())  # [-1, 1] for InceptionV3
    return resized, x[None, ...]


def make_gradcam_model(model, target_layer_name):
    target = model.get_layer(target_layer_name).output
    return tf.keras.Model(inputs=model.input, outputs=[target, model.output])


def gradcam(grad_model, x, class_idx=None):
    x = tf.convert_to_tensor(x)
    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(x, training=False)
        if class_idx is None:
            class_idx = int(tf.argmax(preds[0]))
        target = preds[:, class_idx]
    grads = tape.gradient(target, conv_out)  # (1, H, W, C)
    weights = tf.reduce_mean(grads, axis=(1, 2))  # (1, C)
    cam = tf.reduce_sum(conv_out * weights[:, None, None, :], axis=-1)  # (1, H, W)
    cam = tf.nn.relu(cam)[0].numpy()
    cam_max = cam.max()
    if cam_max > 0:
        cam = cam / cam_max
    return cam, preds[0].numpy(), class_idx


def overlay(rgb_img, cam, alpha=0.4):
    cam_resized = cv2.resize(cam, (rgb_img.shape[1], rgb_img.shape[0]))
    heat = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    return np.uint8((1 - alpha) * rgb_img + alpha * heat), cam_resized


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=2)
    ap.add_argument("--target-layer", default=DEFAULT_TARGET_LAYER)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threshold", type=float, default=0.5, help="for pseudo-mask")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    labels = load_labels()
    samples = pick_samples(labels, per_class=args.per_class, seed=args.seed)
    print(f"Selected {len(samples)} samples ({args.per_class} per class)")

    print("Loading model...")
    model = load_dr_model()
    grad_model = make_gradcam_model(model, args.target_layer)
    print(f"Target conv layer: {args.target_layer} -> {model.get_layer(args.target_layer).output_shape}")

    cols = 4  # original / heatmap overlay / pseudo-mask / masked image
    fig, axes = plt.subplots(len(samples), cols, figsize=(cols * 3.0, len(samples) * 3.0))
    if len(samples) == 1:
        axes = axes[None, :]

    for i, (img_id, true_lvl) in enumerate(samples):
        path = os.path.join(TRAIN_DIR, img_id + ".jpeg")
        rgb, x = load_image(path)
        cam, probs, pred = gradcam(grad_model, x)
        overlayed, cam_up = overlay(rgb, cam)
        mask = (cam_up >= args.threshold).astype(np.uint8) * 255
        masked = rgb.copy()
        masked[mask == 0] = (masked[mask == 0] * 0.25).astype(np.uint8)

        axes[i, 0].imshow(rgb)
        axes[i, 0].set_title(f"{img_id}\ntrue={true_lvl} pred={pred} p={probs[pred]:.2f}", fontsize=8)
        axes[i, 1].imshow(overlayed)
        axes[i, 1].set_title("GradCAM overlay", fontsize=8)
        axes[i, 2].imshow(mask, cmap="gray")
        axes[i, 2].set_title(f"pseudo-mask (t={args.threshold})", fontsize=8)
        axes[i, 3].imshow(masked)
        axes[i, 3].set_title("masked region", fontsize=8)
        for j in range(cols):
            axes[i, j].axis("off")

        print(f"  {img_id}  true={true_lvl} pred={pred}  probs={np.round(probs, 3).tolist()}")

    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, f"gradcam_{args.target_layer}_perclass{args.per_class}.png")
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
