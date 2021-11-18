"""
Testcases for encoders:
- Exclude
- Log10
- Normalization

Separate linear-reduction tests:
- PCA
- Karuhnen-Loeve
"""
import numpy as np
from profit.sur.encoders import Encoder


def test_exclude():

    CONFIG = ['Exclude', [2], False, {}]
    COLUMNS= [2]
    SIZE = (10, 4)
    n = SIZE[0] * SIZE[1]
    X = np.linspace(0, n-1, n).reshape(SIZE)

    enc = Encoder['Exclude'](COLUMNS)
    assert enc.repr == CONFIG
    X_enc = enc.encode(X)
    assert np.all(X_enc == X[:, [0, 1, 3]])
    X_dec = enc.decode(X_enc)
    assert np.all(X_dec == X)


def test_log10():
    from profit.sur.encoders import Log10Encoder

    CONFIG = ['Log10', [2, 3], False, {}]
    COLUMNS= [2, 3]
    SIZE = (10, 4)
    n = SIZE[0] * SIZE[1]
    X = np.linspace(0, n-1, n).reshape(SIZE)
    X_log = X.copy()
    X_log[:, COLUMNS] = np.log10(X_log[:, COLUMNS])

    enc = Log10Encoder(COLUMNS)
    assert enc.repr == CONFIG
    X_enc = enc.encode(X)
    assert np.all(X_enc == X_log)
    X_dec = enc.decode(X_enc)
    assert np.allclose(X_dec, X, atol=1e-7)


def test_normalization():

    CONFIG = ['Normalization', [0, 1, 2, 3], False, {}]
    COLUMNS= [0, 1, 2, 3]
    SIZE = (10, 4)
    n = SIZE[0] * SIZE[1]
    X = np.linspace(0, n-1, n).reshape(SIZE)
    norm_box = np.hstack([np.linspace(0, 1, SIZE[0]).reshape(-1, 1) for _ in range(SIZE[1])])

    enc = Encoder['Normalization'](COLUMNS)
    assert enc.repr == CONFIG
    X_enc = enc.encode(X)
    assert np.allclose(X_enc, norm_box)
    X_dec = enc.decode(X_enc)
    assert np.allclose(X_dec, X)

    y = np.sin(X[:, [1]])
    enc = Encoder['Normalization']([0], work_on_output=True)
    y_enc = enc.encode(y)
    assert max(y_enc) == 1
    assert min(y_enc) == 0
    y_dec = enc.decode(y_enc)
    assert np.allclose(y_dec, y)

    hyperparameters = {'length_scale': np.array([1.3, 0.2, 2.4]),
                       'sigma_f': np.array([0.7]),
                       'sigma_n': np.array([0.05])}

    xmin = 0.5
    xmax = 3
    scaling = xmax - xmin

    for key, value in hyperparameters.items():
        work_on_output = key != 'length_scale'
        enc = Encoder['Normalization'](COLUMNS, work_on_output=work_on_output,
                                       variables={'xmin': 0.5, 'xmax': 3})
        val_dec = enc.decode_hyperparameters(key, value)
        assert np.allclose(val_dec, value * scaling)
