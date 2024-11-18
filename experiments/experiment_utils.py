from spectral_explain.support_recovery import support_recovery
from spectral_explain.qsft.qsft import fit_regression, transform_via_amp
from spectral_explain.qsft.utils import qary_ints_low_order

def linear(signal, b, order=1, **kwargs):
    assert order in [1, 2]
    if order == 1:
        return fit_regression('linear', {'locations': qary_ints_low_order(signal.n, 2, 1).T}, signal, signal.n, b)[0]
    else:
        return fit_regression('linear', {'locations': qary_ints_low_order(signal.n, 2, 2).T}, signal, signal.n, b)[0]


def lasso(signal, b, order=1, **kwargs):
    assert order in [1, 2]

    if order == 1:
        return fit_regression('lasso', {'locations': qary_ints_low_order(signal.n, 2, 1).T}, signal, signal.n, b)[0]
    else:
        return fit_regression('lasso', {'locations': qary_ints_low_order(signal.n, 2, 2).T}, signal, signal.n, b)[0]


def amp(signal, b, order=1, **kwargs):
    assert order in [1,2]
    return transform_via_amp(signal, b, order=order)["transform"]

def qsft_hard(signal, b, t=5, **kwargs):
    return support_recovery("hard", signal, b,t=t)["transform"]


def qsft_soft(signal, b, t=5, **kwargs):
    return support_recovery("soft", signal, b, t=t)["transform"]