# Blind Industrial Image Deblurring via Self-Estimated PSF and Frequency-Domain Restoration

A fully self-contained high-performance **blind image deblurring pipeline** written in **C++ + OpenCV**, designed for restoration of compressed, motion-softened, and mildly defocused images **without requiring any prior blur kernel**.

This project implements a hybrid **spatial-statistical + frequency-correlation Point Spread Function (PSF) estimation**, followed by an adaptive **Wiener-like inverse filtering**, **spectral dequantization**, and **frequency anomaly suppression**.

The algorithm is specifically engineered for:

- JPEG / codec softened images
- motion-smear contaminated luminance
- mild optical blur
- industrial inspection imagery
- document and texture restoration
- fast batch restoration pipelines

Unlike classical deconvolution examples that require a known kernel, this implementation **blindly reconstructs the blur profile directly from the image content itself**.

---

## ✨ Core Features

### ✔ Blind PSF estimation (no user blur kernel required)

The blur kernel is estimated from two independent self-analysis branches:

#### 1. **Maximum Difference Spatial Statistics**
A directional neighborhood differential histogram is accumulated over the image and converted into a probabilistic blur likelihood matrix.

This branch detects:

- dominant blur direction,
- edge spread anisotropy,
- local softness distribution.

Implemented in:

```cpp
computeMaxDiffMatrix()
```

This stage is highly optimized:

-   single-pass neighborhood accumulation,
-   compile-time radius specialization,
-   OpenMP parallel histogram reduction,
-   no dynamic per-pixel allocations.

* * * * *

#### 2\. **FFT Gradient Autocorrelation PSF Detection**

A second PSF estimate is derived from:

-   Sobel gradient energy extraction,
-   gradient thresholding,
-   spectral autocorrelation via FFT,
-   central correlation crop.

Implemented in:

computeCorrelationFFT()

This branch is especially sensitive to:

-   repeated edge spread,
-   directional motion signatures,
-   latent defocus halo.

* * * * *

### ✔ Hybrid PSF fusion

Both independent blur estimators are fused:

M = computeMaxDiffMatrix(...) + computeCorrelationFFT(...);

This produces a much more stable PSF candidate than either branch alone.

* * * * *

### ✔ Automatic PSF cleanup and active region extraction

The raw PSF estimate is further refined by:

-   percentile heap clipping,
-   zero-energy border removal,
-   active support extraction,
-   unit-sum normalization.

Implemented in:

buildPSFFromM()

This eliminates diffuse low-confidence tails and preserves only the blur-support core.

* * * * *

### ✔ Adaptive inverse filter synthesis

The final PSF is converted into a frequency-domain inverse restoration filter using an adaptive stabilized Wiener denominator:

G(u,v) = H*(u,v) / (|H(u,v)|² + K)

with automatic scaling of `K` according to the mean spectral energy of the kernel.

Implemented in:

buildInverseFilterFromPSF()

This avoids catastrophic singular amplification common in naïve inverse deconvolution.

* * * * *

### ✔ Fourier-domain dequantization compensation

Most blurred images distributed through JPEG/video pipelines suffer not only from blur but also from:

-   DCT coefficient rounding,
-   quantization noise floor,
-   low-amplitude spectral attenuation.

This project compensates for that by applying a statistically derived magnitude shrinkage correction directly in Fourier space.

Implemented in:

DeQuantization()

This step significantly improves:

-   fine edge recovery,
-   text clarity,
-   microtexture retention.

* * * * *

### ✔ Spectral anomaly suppression

Inverse filtering often explosively amplifies:

-   ringing frequencies,
-   JPEG block harmonics,
-   codec mosquito noise,
-   isolated FFT spikes.

This implementation contains a custom industrial anomaly suppressor that:

1.  computes log-spectrum residuals,
2.  subtracts a Gaussian spectral baseline,
3.  z-score isolates anomalous peaks,
4.  applies radial mid-frequency weighting,
5.  softly attenuates unstable frequencies.

Implemented in:

AnomalySuppression()

This makes the restoration visually stable and much less noisy than standard Wiener examples.

* * * * *

### ✔ Luminance-only restoration workflow

The algorithm works in:

BGR → YCrCb → Deblur Y only → Restore color

Meaning:

-   luminance is sharpened aggressively,
-   chroma remains untouched,
-   color artifacts are minimized.

This preserves natural color reproduction while maximizing detail gain.

* * * * *

⚙ Processing Pipeline Overview
------------------------------

Input Image\
   │\
   ├─ Convert to YCrCb\
   │\
   ├─ Deblur luminance channel:\
   │     ├─ Spatial max-diff PSF estimate\
   │     ├─ FFT correlation PSF estimate\
   │     ├─ PSF fusion\
   │     ├─ PSF cleanup\
   │     ├─ Inverse filter construction\
   │     ├─ Forward FFT of image\
   │     ├─ Dequantization compensation\
   │     ├─ Spectral anomaly suppression\
   │     ├─ Inverse deconvolution\
   │\
   └─ Recombine chroma and save output

* * * * *

🚀 Performance Characteristics
------------------------------

Designed for industrial-scale usage:

-   heavy loops are OpenMP parallelized,
-   histogram accumulation is cache-friendly,
-   FFT operations use OpenCV optimized backend,
-   all matrices remain `CV_32F`,
-   no third-party dependencies beyond OpenCV.

Typical processing time on modern desktop CPU:

-   ~0.3s to 1.5s for HD images depending on radius and FFT size.

* * * * *

🧠 Why this project is different
--------------------------------

Most public "deblur" examples are one of the following:

-   Richardson-Lucy demos,
-   naïve Wiener filter wrappers,
-   manually supplied Gaussian kernels,
-   neural models requiring training.

This repository is different:

### It is:

-   deterministic,
-   explainable,
-   training-free,
-   blind,
-   fully CPU-native,
-   industrially embeddable.

No AI weights.\
No external models.\
No user blur parameters.

Just pure signal processing.

* * * * *

📦 Build Requirements
---------------------

-   C++17 or newer
-   OpenCV 4.x
-   OpenMP enabled compiler

### Linux / GCC example

g++ main.cpp -O3 -march=native -fopenmp `pkg-config --cflags --libs opencv4` -o blind_deblur

### Windows / MSVC

Enable:

-   `/O2`
-   `/openmp`

and link against OpenCV.

* * * * *

▶ Usage
-------

blind_deblur input.jpg output.jpg

Optional third parameter saves the restored luminance channel for inspection:

blind_deblur input.jpg output.jpg luminance_debug.jpg

* * * * *

📁 Main Important Functions
---------------------------

| Function | Purpose |
| --- | --- |
| `computeMaxDiffMatrix()` | spatial statistical PSF inference |
| `computeCorrelationFFT()` | FFT autocorrelation PSF inference |
| `buildPSFFromM()` | PSF cleanup and support extraction |
| `buildInverseFilterFromPSF()` | adaptive inverse filter generation |
| `DeQuantization()` | JPEG/codec quantization compensation |
| `AnomalySuppression()` | unstable spectral peak damping |
| `deblurChannel()` | full luminance restoration pipeline |

* * * * *

🔬 Best suited for
------------------

-   scanned documents
-   surveillance frames
-   industrial camera output
-   microscopy snapshots
-   compressed web imagery
-   archival photo enhancement

* * * * *

⚠ Limitations
-------------

This is not a miracle neural hallucination enhancer.

It works best on:

-   mild to moderate blur,
-   directional softness,
-   codec-induced loss.

Extremely severe defocus or large nonlinear motion blur may require iterative or multi-kernel extensions.

* * * * *

📜 License
----------

MIT License (or choose your preferred license)
