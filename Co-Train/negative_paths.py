# negative_paths.py -- how often the delete-the-path policy fires, per
# first-hit round, for a grid of sd at a0 = 1.25 and rounds + 1 = 5 draws
import numpy as np
A0, DRAWS, N = 1.25, 5, 1_000_000
rng = np.random.default_rng(0)
for sd in (0.05, 0.10, 0.20, 0.30, 0.40):
    rates = A0 + sd * np.cumsum(rng.standard_normal((N, DRAWS)), axis=1)
    touched = rates <= 0
    hit = touched.any(axis=1)
    first = np.where(hit, touched.argmax(axis=1) + 1, 0)
    print(f"sd {sd:.2f}  P(deleted) {hit.mean():.2e}  by first-hit draw "
          + "  ".join(f"{(first == r).mean():.1e}" for r in range(1, DRAWS + 1)))