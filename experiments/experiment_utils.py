from spectral_explain.support_recovery import support_recovery
from spectral_explain.qsft.qsft import fit_regression
from spectral_explain.qsft.utils import qary_ints_low_order
from spectral_explain.utils import mobius_to_fourier, fourier_to_mobius

def lime(signal, b, **kwargs):
    return fit_regression('lasso', {'locations': qary_ints_low_order(signal.n, 2, 1).T}, signal, signal.n, b)[0]
def linear(signal, b, order=1, **kwargs):
    return fit_regression('linear', {'locations': qary_ints_low_order(signal.n, 2, order).T}, signal, signal.n, b)[0]

def lasso(signal, b, order=1, **kwargs):
    return fit_regression('lasso', {'locations': qary_ints_low_order(signal.n, 2, order).T}, signal, signal.n, b)[0]

def faith_banzhaf(signal, b, order=1, **kwargs):
    return fit_regression('lasso', {'locations': qary_ints_low_order(signal.n, 2, order).T}, signal, signal.n, b,
                          fourier_basis=False)[0]

def faith_shapley(signal, b, order=1, **kwargs):
    return fit_regression('shapley', {'locations': qary_ints_low_order(signal.n, 2, order).T}, signal, signal.n, b,
                          fourier_basis=False)[0]

def qsft_hard(signal, b, t=5, **kwargs):
    return support_recovery("hard", signal, b, t=t)["transform"]


def qsft_soft(signal, b, t=5, **kwargs):
    return support_recovery("soft", signal, b, t=t)["transform"]
