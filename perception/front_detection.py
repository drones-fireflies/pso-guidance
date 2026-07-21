import cv2
import numpy as np


def cv_frontier(fire: np.ndarray, blur_threshold: int = 60) -> np.ndarray:
    """
    OpenCV-based fire front detection.

    Parameters
    ----------
    fire            : where > 0 means burning.
    blur_threshold  : binarisation threshold after blurring.
                      Lower  → frontier a bit larger, fewer holes
                      Higher → tighter frontier, possible internal holes

    """

    fire = fire.astype(np.uint8)

    blur = cv2.blur(fire, (5, 5))
    _, blur = cv2.threshold(blur, blur_threshold, 255, cv2.THRESH_BINARY)
    
    blur = cv2.medianBlur(blur, 5)
    combined = np.clip(blur + fire, 0, 255).astype(np.uint8)

    # one extra dilation pass
    blur = cv2.blur(combined, (5, 5))
    _, blur = cv2.threshold(blur, blur_threshold, 255, cv2.THRESH_BINARY)
    blur = cv2.medianBlur(blur, 5)
    combined = np.clip(blur + combined, 0, 255).astype(np.uint8)

    combined = cv2.GaussianBlur(combined, (9, 9), 1.5)
    edges = cv2.Canny(combined, threshold1=150, threshold2=200)
    return edges


def frontier_cells(fire_uint8: np.ndarray, blur_threshold: int = 70) -> np.ndarray:
    """Return (N,2) array of frontier pixel coordinates [row, col]."""
    edges = cv_frontier(fire_uint8, blur_threshold)
    pts = np.argwhere(edges > 0)
    return pts