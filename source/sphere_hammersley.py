"""Low-discrepancy camera placement on a sphere.

Extracted from TRELLIS dataset_toolkits/utils.py for use outside Blender.
"""

import hashlib

import numpy as np

PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53]


def radical_inverse(base: int, n: int) -> float:
    val = 0.0
    inv_base = 1.0 / base
    inv_base_n = inv_base
    while n > 0:
        digit = n % base
        val += digit * inv_base_n
        n //= base
        inv_base_n *= inv_base
    return val


def halton_sequence(dim: int, n: int) -> list[float]:
    return [radical_inverse(PRIMES[d], n) for d in range(dim)]


def hammersley_sequence(dim: int, n: int, num_samples: int) -> list[float]:
    return [n / num_samples] + halton_sequence(dim - 1, n)


def sphere_hammersley_sequence(
    n: int, num_samples: int, offset: tuple[float, float] = (0, 0)
) -> tuple[float, float]:
    u, v = hammersley_sequence(2, n, num_samples)
    u += offset[0] / num_samples
    v += offset[1]
    u = 2 * u if u < 0.25 else 2 / 3 * u + 1 / 3
    theta = float(np.arccos(1 - 2 * u) - np.pi / 2)
    phi = float(v * 2 * np.pi)
    return phi, theta


def generate_views(
    num_views: int,
    radius: float = 2.0,
    fov_deg: float = 40.0,
    uid: str | None = None,
) -> list[dict]:
    """Generate camera views for Blender rendering.

    Args:
        num_views: Number of views to generate.
        radius: Camera distance from origin.
        fov_deg: Field of view in degrees.
        uid: If provided, used as seed for the per-object random offset
             (ensures reproducibility). Otherwise uses numpy random.
    """
    if uid is not None:
        seed = int(hashlib.sha256(uid.encode()).hexdigest()[:8], 16)
        rng = np.random.RandomState(seed)
        offset = (rng.rand(), rng.rand())
    else:
        offset = (np.random.rand(), np.random.rand())

    fov_rad = fov_deg / 180 * np.pi
    views = []
    for i in range(num_views):
        yaw, pitch = sphere_hammersley_sequence(i, num_views, offset)
        views.append({
            "yaw": yaw,
            "pitch": pitch,
            "radius": radius,
            "fov": float(fov_rad),
        })
    return views
