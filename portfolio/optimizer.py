import numpy as np

def efficient_frontier(returns):

    mean_returns = returns.mean() * 252
    cov = returns.cov() * 252

    results = []

    for _ in range(3000):
        weights = np.random.random(len(mean_returns))
        weights /= np.sum(weights)

        ret = np.sum(mean_returns * weights)
        vol = np.sqrt(np.dot(weights.T, np.dot(cov, weights)))
        sharpe = ret / vol

        results.append([ret, vol, sharpe])

    return np.array(results)
