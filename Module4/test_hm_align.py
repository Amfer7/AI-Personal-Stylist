import numpy as np

from hm_catalogue import coral_transform, shift_to_means


def _cloud(rng, n, d, centre, scale):
    return centre + rng.standard_normal((n, d)) * scale


def test_coral_matches_target_mean_and_covariance_per_type():
    rng = np.random.default_rng(0)
    d = 8
    # centred clouds stay full-rank after unit normalisation (an off-centre cloud loses its radial axis)
    src = np.concatenate([_cloud(rng, 4000, d, 0.0, 1.0), _cloud(rng, 4000, d, 0.0, np.linspace(0.5, 2, d))])
    types = np.repeat([0, 1], 4000)
    A = rng.standard_normal((d, d)) * 0.1
    means = rng.standard_normal((2, d))
    covs = np.stack([A @ A.T, 0.01 * np.eye(d)])
    out = coral_transform(src, types, means, covs, renorm=False)
    for t in (0, 1):
        x = out[types == t]
        assert np.allclose(x.mean(0), means[t], atol=1e-6)
        assert np.allclose(np.cov(x, rowvar=False), covs[t], atol=0.03 * np.abs(covs[t]).max())


def test_coral_leaves_untyped_rows_and_returns_unit_vectors():
    rng = np.random.default_rng(1)
    src = rng.standard_normal((600, 6))
    types = np.repeat([0, -1], 300)
    out = coral_transform(src, types, rng.standard_normal((1, 6)), np.eye(6)[None] * 0.1)
    assert np.allclose(np.linalg.norm(out, axis=1), 1.0, atol=1e-5)
    unit = src / np.linalg.norm(src, axis=1, keepdims=True)
    assert np.allclose(out[300:], unit[300:], atol=1e-6)


def test_mean_shift_still_matches_means():
    rng = np.random.default_rng(2)
    src = rng.standard_normal((500, 4)) + 3
    types = np.zeros(500, np.int64)
    out = shift_to_means(src, types, np.array([[0.5, 0.5, 0.5, 0.5]]))
    assert np.allclose(np.linalg.norm(out, axis=1), 1.0, atol=1e-5)
