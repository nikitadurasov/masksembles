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

    if c < 10:
        raise ValueError("Masksembles approach couldn't be used in such setups where "
                         f"number of channels is less then 10. Current value is (channels={c}). "
                         "Please increase number of features in your layer or remove this "
                         "particular instance of Masksembles from your architecture.")

    if scale > 6.:
        raise ValueError("Masksembles approach couldn't be used in such setups where "
                         f"scale parameter is larger then 6. Current value is (scale={scale}).")

    # inverse formula for number of active features in masks
    active_features = int(int(c) / (scale * (1 - (1 - 1 / scale) ** n)))

    # FIXME this piece searches for scale parameter value that generates
    #  proper number of features in masks, sometimes search is not accurate
    #  enough and masks.shape != c. Could fix it with binary search.
    masks = generate_masks(active_features, n, scale)
    for s in np.linspace(max(0.8 * scale, 1.0), 1.5 * scale, 300):
        if masks.shape[-1] >= c:
            break
        masks = generate_masks(active_features, n, s)
    new_upper_scale = s

    if masks.shape[-1] != c:
        for s in np.linspace(max(0.8 * scale, 1.0), new_upper_scale, 1000):
            if masks.shape[-1] >= c:
                break
            masks = generate_masks(active_features, n, s)

    if masks.shape[-1] != c:
        raise ValueError("generation_wrapper function failed to generate masks with "
                         "requested number of features. Please try to change scale parameter")

    return masks
