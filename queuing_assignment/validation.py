import numpy as np
from scipy import stats
from random_generators import exponential, poisson


def validate_exponential(mean=10.0, n=100_000):
    print("=" * 50)
    print(f"Exponential validation  (E[X]={mean}, N={n:,})")
    print("=" * 50)

    custom = np.array(exponential(mean=mean, size=n))
    scipy_ = stats.expon.rvs(scale=mean, size=n)

    # KS test: custom samples vs theoretical distribution
    ks_stat, ks_p = stats.kstest(custom, 'expon', args=(0, mean))

    print(f"  Custom mean       : {custom.mean():.4f}  (expected {mean})")
    print(f"  Custom std        : {custom.std():.4f}   (expected {mean})")
    print(f"  Scipy  mean       : {scipy_.mean():.4f}")
    print(f"  KS statistic      : {ks_stat:.6f}")
    print(f"  KS p-value        : {ks_p:.4f}  {'PASS' if ks_p > 0.05 else 'FAIL'}")
    print()

    return custom


def validate_poisson(mean=10.0, n=100_000):
    print("=" * 50)
    print(f"Poisson validation  (E[Z]={mean}, N={n:,})")
    print("=" * 50)

    custom = np.array(poisson(mean=mean, size=n))
    scipy_ = stats.poisson.rvs(mu=mean, size=n)

    # KS test on empirical CDF
    ks_stat, ks_p = stats.kstest(custom, stats.poisson(mean).cdf)

    print(f"  Custom mean       : {custom.mean():.4f}  (expected {mean})")
    print(f"  Custom var        : {custom.var():.4f}   (expected {mean})")
    print(f"  Scipy  mean       : {scipy_.mean():.4f}")
    print(f"  KS statistic      : {ks_stat:.6f}")
    print(f"  KS p-value        : {ks_p:.4f}  {'PASS' if ks_p > 0.05 else 'FAIL'}")
    print()

    return custom


if __name__ == "__main__":
    validate_exponential()
    validate_poisson()
