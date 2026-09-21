

import cv2
import matplotlib.pyplot as plt
import numpy as np


def analyze_edges(img_rgb: np.ndarray) -> np.ndarray:
    """Retourne une image en niveaux de gris des contours détectés (Canny)."""
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 100, 200)
    return edges


def analyze_sharpness(img_rgb: np.ndarray) -> float:
    """Calcule un score de netteté via la variance du Laplacien."""
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def color_histogram(img_rgb: np.ndarray):
    """Construit un histogramme des canaux R, G, B et le retourne (figure matplotlib)."""
    fig, ax = plt.subplots(figsize=(6, 3))
    colors = ("r", "g", "b")
    for i, color in enumerate(colors):
        hist = cv2.calcHist([img_rgb], [i], None, [256], [0, 256])
        ax.plot(hist, color=color, linewidth=1)
    ax.set_xlim([0, 256])
    ax.set_xlabel("Intensité du pixel")
    ax.set_ylabel("Nombre de pixels")
    ax.set_title("Histogramme des couleurs")
    fig.tight_layout()
    return fig


def dominant_colors(img_rgb: np.ndarray, k: int = 5):
    """Extrait les k couleurs dominantes de l'image via K-means."""
    data = img_rgb.reshape((-1, 3)).astype(np.float32)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, labels, centers = cv2.kmeans(
        data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS
    )

    # Trie les couleurs par fréquence d'apparition (la plus fréquente en premier)
    counts = np.bincount(labels.flatten(), minlength=k)
    order = np.argsort(-counts)
    centers = centers[order].astype(int)

    return [tuple(c) for c in centers]
