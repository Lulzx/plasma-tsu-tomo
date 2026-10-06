# Chord-variance inflation vs K_z (evidence lambda, compensation on, tau/dz=0.75)

inflation = mean (s^2 + sum tau^2)/sigma^2 - 1 over chords; clamped_frac = chords where sum tau^2 >= sigma^2 so compensation cannot be exact.

| phantom | variant | Kz | n_spins | n_blocks | max_degree | inflation | clamped_frac |
|---|---|---|---|---|---|---|---|
| peaked | chain | 16 | 38852 | 44 | 364 | 26.5 | 1 |
| peaked | tree | 16 | 37772 | 66 | 244 | 12.1 | 1 |
| peaked | chain | 32 | 74276 | 76 | 716 | 5.51 | 1 |
| peaked | tree | 32 | 72044 | 114 | 468 | 2.15 | 1 |
| peaked | chain | 64 | 145124 | 140 | 1420 | 0.719 | 0.667 |
| peaked | tree | 64 | 140588 | 210 | 916 | 0.121 | 0.333 |
| peaked | chain | 128 | 286820 | 268 | 2828 | 0.005 | 0.056 |
| peaked | tree | 128 | 277676 | 402 | 1812 | 0 | 0 |
| hollow | chain | 16 | 38852 | 44 | 364 | 20.4 | 1 |
| hollow | tree | 16 | 37772 | 66 | 244 | 9.31 | 1 |
| hollow | chain | 32 | 74276 | 76 | 716 | 4.09 | 1 |
| hollow | tree | 32 | 72044 | 114 | 468 | 1.49 | 1 |
| hollow | chain | 64 | 145124 | 140 | 1420 | 0.363 | 0.653 |
| hollow | tree | 64 | 140588 | 210 | 916 | 0.039 | 0.153 |
| hollow | chain | 128 | 286820 | 268 | 2828 | -0 | 0 |
| hollow | tree | 128 | 277676 | 402 | 1812 | 0 | 0 |

