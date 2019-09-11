import numpy as np
import scipy.stats as stats

# Statistical significance test for comparison between classifiers

def wilsonscore(metric, N=0, p=0.05):
    assert N > 0, 'Number of sample should not be zero'
    assert p in [0.1, 0.05, 0.02, 0.01], 'p value should be in [0.1, 0.05, 0.02, 0.01]'

    if p == 0.1:
        z = 1.64
    elif p == .05:
        z = 1.96
    elif p == 0.02:
        z = 2.33
    elif p == 0.01:
        z = 2.58

    A = N/(N+z**2)
    B = metric + z**2/(2*N)
    C = z*np.sqrt(
        metric*(1.-metric)/N + z**2/(4*N**2)
    )

    low = A*(B-C)
    high = A*(B+C)
    return np.array([low, high])