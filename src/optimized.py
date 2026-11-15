"""Small, unpenalized logistic MLE with safeguarded Newton steps.

No covariance matrix is needed by this study. Design has three columns.
Nonexistent, singular or unconverged fits are explicit failures, not estimates.
"""
import numpy as np
from scipy.special import expit, logsumexp


def logodds(p):
    return np.log(p) - np.log1p(-p) if 0 < p < 1 else np.nan


def estimate_effects(X, Z, Y_obs, diagnostics=False):
    X, Z, y = (np.asarray(a, dtype=float) for a in (X, Z, Y_obs))
    if X.ndim != 1 or X.shape != Z.shape or X.shape != y.shape or not len(X):
        raise ValueError('Expected three nonempty vectors of equal length')
    if not all(np.isfinite(a).all() for a in (X, Z, y)):
        raise ValueError('Inputs must be finite')
    if not np.isin(X, [0, 1]).all() or not np.isin(y, [0, 1]).all():
        raise ValueError('Treatment and outcome must be binary')
    nt = X.sum(); nc = len(X)-nt
    itt = logodds(y[X == 1].mean())-logodds(y[X == 0].mean()) if nt and nc else np.nan
    info = {'status': 'ok', 'iterations': 0, 'itt_boundary': bool(not np.isfinite(itt))}
    def done(b=np.nan, p=np.nan):
        value = (b, p, itt)
        return (value, info) if diagnostics else value
    zmean, zscale = Z.mean(), Z.std()
    if nt == 0 or nc == 0 or zscale == 0:
        info['status'] = 'singular_design'; return done()
    if y.min() == y.max():
        info['status'] = 'constant_outcome'; return done()
    z = (Z-zmean)/zscale
    A = np.column_stack((np.ones(len(X)), X, z))
    beta = np.zeros(3)
    for iteration in range(1, 61):
        eta = A @ beta
        p = expit(eta)
        # expit(-eta) avoids cancellation in 1-expit(eta).
        w = p * expit(-eta)
        H = A.T @ (w[:, None]*A)
        g = A.T @ (p-y)
        info['iterations'] = iteration
        if not np.isfinite(H).all() or np.linalg.cond(H) > 1e12:
            info['status'] = 'ill_conditioned'; return done()
        step = np.linalg.solve(H, g)
        loss = np.sum(np.logaddexp(0, eta)-y*eta)
        scale = 1.
        while scale >= 2**-20:
            trial = beta-scale*step
            eta_trial = A @ trial
            new_loss = np.sum(np.logaddexp(0, eta_trial)-y*eta_trial)
            if np.isfinite(new_loss) and new_loss <= loss + 1e-12:
                break
            scale *= .5
        else:
            info['status'] = 'line_search_failed'; return done()
        beta = trial
        # Require both small step and score; small score alone can accept separation.
        if np.max(np.abs(scale*step)) < 1e-9 and np.max(np.abs(A.T @ (expit(eta_trial)-y)))/len(y) < 1e-9:
            break
    else:
        info['status'] = 'nonconverged'; return done()
    eta_c = beta[0]+beta[2]*z
    eta_t = eta_c+beta[1]
    # log(mean(p))-log(mean(1-p)); log(n) cancels in each log odds.
    def marginal_logodds(eta):
        return logsumexp(-np.logaddexp(0, -eta))-logsumexp(-np.logaddexp(0, eta))
    plugin = marginal_logodds(eta_t)-marginal_logodds(eta_c)
    return done(beta[1], plugin)
