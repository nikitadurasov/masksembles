import numpy as np


def generate_masks_(m: int, n: int, s: float) -> np.ndarray:
    """Generates set of binary masks with properties defined by n, m, s params.

    Results of this function are stochastic, that is, calls with the same sets
    of arguments might generate outputs of different shapes. Check generate_masks
    and generation_wrapper function for more deterministic behaviour.

    :param m: int, number of ones in each mask
    :param n: int, number of masks in the set
    :param s: float, scale param controls overlap of generated masks
    :return: np.ndarray, matrix of binary vectors
    """

    total_positions = int(m * s)
    masks = []

    for _ in range(n):
        new_vector = np.zeros([total_positions])
        idx = np.random.choice(range(total_positions), m, replace=False)
        new_vector[idx] = 1
        masks.append(new_vector)

    masks = np.array(masks)
    # drop useless positions
    masks = masks[:, ~np.all(masks == 0, axis=0)]
    return masks


def generate_masks(m: int, n: int, s: float) -> np.ndarray:
    """Generates set of binary masks with properties defined by n, m, s params.

    Resulting masks are required to have fixed features size as it's described in [1].
    Since process of masks generation is stochastic therefore function evaluates
    generate_masks_ multiple times till expected size is acquired.

    :param m: int, number of ones in each mask
    :param n: int, number of masks in the set
    :param s: float, scale param controls overlap of generated masks
    :return: np.ndarray, matrix of binary vectors

    References

    [1] `Masksembles for Uncertainty Estimation: Supplementary Material`,
    Nikita Durasov, Timur Bagautdinov, Pierre Baque, Pascal Fua
    """

    masks = generate_masks_(m, n, s)
    # hardcoded formula for expected size, check reference
    expected_size = int(m * s * (1 - (1 - 1 / s) ** n))
    while masks.shape[1] != expected_size:
        masks = generate_masks_(m, n, s)
    return masks


def generation_wrapper(c: int, n: int, scale: float) -> np.ndarray:
    """Generates set of binary masks with properties defined by c, n, scale params.

     Allows to generate masks sets with predefined features number c. Particularly
     convenient to use in torch-like layers where one need to define shapes inputs
     tensors beforehand.

    :param c: int, number of channels in generated masks
    :param n: int, number of masks in the set
    :param scale: float, scale param controls overlap of generated masks
    :return: np.ndarray, matrix of binary vectors
    """
    active_features = int(c / (scale * (1 - (1 - 1 / scale) ** n)))
    for s in np.linspace(1, 3 / 2 * scale, 100):
        masks = generate_masks(active_features, n, s)
        if masks.shape[-1] == c:
            break
    assert masks.shape[-1] == c, "Failed to generate proper masks, try other value for scale"
    return masks
