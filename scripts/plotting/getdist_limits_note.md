# Why `marge.texValues` can differ from `twoTailLimits` and `mean`

## Short answer

- **`samples[0].mean('w0')`** and **`marge.parWithName('w0').mean`** are the same: both use the **sample mean** of the chain.
- **`samples[0].twoTailLimits('w0', 0.68)`** always uses **equal-tailed** limits: 16th and 84th **sample percentiles** (raw chain).
- **`marge.texValues(..., limit=1)`** uses limits from **`getMargeStats()`**, which are **not always** equal-tailed. GetDist often uses limits from the **1D kernel density estimate (KDE)** instead of sample percentiles.

So the mean agrees; the intervals can disagree when GetDist chooses KDE-based limits.

## What GetDist does in `getMargeStats()` (1D limits)

1. Builds a 1D **smoothed density** (KDE) for each parameter.
2. For each contour (e.g. 68%, 95%), it gets an interval from that KDE (smallest interval containing that probability under the KDE — similar to a highest-posterior-density interval).
3. It also computes **equal-tailed** limits from the **raw samples** (same idea as `twoTailLimits`).
4. It **only uses the sample percentiles** when the density at the lower and upper percentiles is similar:
   - If `|density(84th %ile) - density(16th %ile)| < credible_interval_threshold` (default 0.05), it uses the **equal-tailed** (sample percentile) limits.
   - Otherwise it keeps the **KDE-based** limits.

So for **skewed** posteriors, `marge.texValues` can show different bounds than `twoTailLimits`, even though both are valid 68% (or 95%) credible intervals.

## How to make them agree (force equal-tailed)

Force GetDist to always use equal-tailed limits (sample percentiles) by setting `credible_interval_threshold` on the **MCSamples** instance to a large value (e.g. 1.0). If you have already called `getMargeStats()` once, clear the cache so limits are recomputed:

```python
# Force equal-tailed limits so marge matches twoTailLimits
samples[0].credible_interval_threshold = 1.0
samples[0].done_1Dbins = False   # invalidate cache so limits are recomputed
marge = samples[0].getMargeStats()
```

Then `marge.texValues(formatter, 'w0', limit=1)` and `samples[0].twoTailLimits('w0', 0.68)` will refer to the same interval (and similarly for limit=2 and 0.95).

## Summary

| Quantity              | Source                          |
|----------------------|----------------------------------|
| Mean                 | Same: sample mean                |
| twoTailLimits(0.68)  | Always: 16th & 84th percentiles |
| marge limits (68%)   | Usually KDE-based; if symmetric, sample percentiles |

To get identical numbers, set `credible_interval_threshold = 1.0` so GetDist always uses the sample percentiles.
