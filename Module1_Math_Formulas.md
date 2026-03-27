# Module-1 Mathematical Formulation (Spatial Feature Extraction)

This document lists the equations used in Module-1 and defines all symbols.

## 1) Multi-modal landmark fusion (per frame)
For frame \(t\):

\[
\mathbf{f}_t = [\mathbf{p}_t \; || \; \mathbf{fa}_t \; || \; \mathbf{lh}_t \; || \; \mathbf{rh}_t]
\]

Where:
- \(\mathbf{f}_t\): fused feature vector for frame \(t\)
- \(\mathbf{p}_t\): pose feature block
- \(\mathbf{fa}_t\): face feature block
- \(\mathbf{lh}_t\): left-hand feature block
- \(\mathbf{rh}_t\): right-hand feature block
- \(||\): concatenation operator

## 2) Feature dimension equation

\[
D = 3\,(N_p + N_f + N_{lh} + N_{rh})
\]

In our implementation:

\[
D = 3\,(25 + 70 + 21 + 21)=411
\]

Where:
- \(D\): final feature dimension per frame
- \(N_p\): selected pose landmarks (25)
- \(N_f\): selected face landmarks (70)
- \(N_{lh}\): selected left-hand landmarks (21)
- \(N_{rh}\): selected right-hand landmarks (21)
- factor 3: each landmark contributes \((x,y,z)\)

## 3) Raw Holistic landmark count (reference)

\[
N_{raw} = 33 + 468 + 21 + 21 = 543
\]

Where:
- 33 pose landmarks
- 468 face landmarks
- 21 landmarks per hand

## 4) Temporal sequence construction
For sequence length \(T\):

\[
\mathbf{X} =
\begin{bmatrix}
\mathbf{f}_1 \\
\mathbf{f}_2 \\
\vdots \\
\mathbf{f}_T
\end{bmatrix}
\in \mathbb{R}^{T \times D}
\]

In our setup:

\[
\mathbf{X} \in \mathbb{R}^{30 \times 411}
\]

Where:
- \(T\): sequence length (usually 30)
- \(D\): feature dimension (411)

## 5) Uniform temporal frame sampling
If a sign segment has \(N\) frames and target sequence length is \(T\), sampled frame indices are:

\[
i_k = \left\lfloor \frac{k\,(N-1)}{T-1} \right\rfloor, \quad k=0,1,\dots,T-1
\]

Equivalent to `linspace(0, N-1, T)` behavior.

Where:
- \(N\): available frames in sign segment
- \(T\): required sampled frames
- \(i_k\): sampled frame index at step \(k\)

## 6) Sequence-wise mean (per feature column)

\[
\mu_j = \frac{1}{T}\sum_{t=1}^{T} X_{t,j}
\]

Where:
- \(j\): feature dimension index (1 to 411)
- \(X_{t,j}\): value of feature \(j\) at time step \(t\)

## 7) Sequence-wise standard deviation (per feature column)

\[
\sigma_j = \sqrt{\frac{1}{T}\sum_{t=1}^{T}(X_{t,j}-\mu_j)^2}
\]

## 8) Normalization (z-score with stability term)

\[
\hat{X}_{t,j} = \frac{X_{t,j}-\mu_j}{\sigma_j+\epsilon}
\]

Where:
- \(\hat{X}_{t,j}\): normalized value
- \(\epsilon = 10^{-6}\): small constant to avoid division by zero

## 9) Missing landmark handling (zero-fill)
If a landmark/modality is not detected in a frame:

\[
(x,y,z) \leftarrow (0,0,0)
\]

This guarantees fixed-length vectors and stable downstream batching.

## 10) Module-1 output contract
- Per frame output: \(\mathbf{f}_t \in \mathbb{R}^{411}\)
- Per sequence output: \(\mathbf{X} \in \mathbb{R}^{30\times411}\)

This normalized sequence is passed to Module-2 (LSTM/BiLSTM).
