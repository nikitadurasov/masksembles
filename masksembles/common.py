import numpy as np


def generate_masks_(m: int, n: int, s: float) -> np.array:
    total_positions = int(m * s)
    masks = []

    for _ in range(n):
        new_vector = np.zeros([total_positions])
        idx = np.random.choice(range(total_positions), m, replace=False)
        new_vector[idx] = 1
        masks.append(new_vector)

    masks = np.array(masks)
    masks = masks[:, ~np.all(masks == 0, axis=0)]
    return masks


def generate_masks(m: int, n: int, s: float) -> np.array:
    masks = generate_masks_(m, n, s)
    expected_size = int(m * s * (1 - (1 - 1 / s) ** n))
    while masks.shape[1] != expected_size:
        masks = generate_masks_(m, n, s)
    return masks


def generation_wrapper(c: int, n: int, scale: float) -> np.array():
    M = int(c / (scale * (1 - (1 - 1 / scale) ** n)))
    for s in np.linspace(1, 3 / 2 * scale, 100):
        masks = generate_masks(M, n, s)
        if masks.shape[-1] == c:
            break
    assert masks.shape[-1] == c, "Failed to generate proper masks, try other value for scale"
    return masks
