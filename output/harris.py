import numpy as np


def get_harris_corners(im: np.ndarray, edge_discard: int = 20):
    assert edge_discard >= 20

    sobel_x = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=np.float32) / 8.0
    sobel_y = sobel_x.T

    def convolve(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        kh, kw = kernel.shape
        pad_y, pad_x = kh // 2, kw // 2
        padded = np.pad(img, ((pad_y, pad_y), (pad_x, pad_x)), mode="reflect")
        windows = np.lib.stride_tricks.sliding_window_view(padded, (kh, kw))
        return np.einsum("ij,xyij->xy", kernel, windows).astype(np.float32)

    ix = convolve(im, sobel_x)
    iy = convolve(im, sobel_y)

    gauss = gaussian_kernel(5, sigma=5 / 3)
    ix2 = convolve(ix * ix, gauss)
    iy2 = convolve(iy * iy, gauss)
    ixy = convolve(ix * iy, gauss)

    k = 0.04
    det = ix2 * iy2 - ixy**2
    trace = ix2 + iy2
    h = det - k * (trace**2)
    h = np.where(h > 0, h, 0.0)

    threshold = max(0.005 * h.max(), 1e-6)
    response = h.copy()
    response[h < threshold] = 0

    coords = np.argwhere(response > 0)
    if coords.size == 0:
        return h, np.empty((2, 0), dtype=np.int64)

    mask = (
        (coords[:, 0] > edge_discard)
        & (coords[:, 0] < im.shape[0] - edge_discard)
        & (coords[:, 1] > edge_discard)
        & (coords[:, 1] < im.shape[1] - edge_discard)
    )
    coords = coords[mask]
    return h, coords.T


def gaussian_kernel(size: int, sigma: float) -> np.ndarray:
    if size % 2 == 0:
        raise ValueError("Gaussian kernel size must be odd")
    ax = np.arange(-(size // 2), size // 2 + 1, dtype=np.float32)
    kernel_1d = np.exp(-0.5 * (ax / sigma) ** 2)
    kernel_1d /= kernel_1d.sum()
    return np.outer(kernel_1d, kernel_1d).astype(np.float32)


def dist2(x: np.ndarray, c: np.ndarray) -> np.ndarray:
    ndata, dimx = x.shape
    ncenters, dimc = c.shape
    if dimx != dimc:
        raise ValueError("Data dimension does not match dimension of centers")

    return (
        (np.ones((ncenters, 1)) * np.sum((x**2).T, axis=0)).T
        + np.ones((ndata, 1)) * np.sum((c**2).T, axis=0)
        - 2 * np.inner(x, c)
    )
