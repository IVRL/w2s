# W2S: A Joint Denoising and Super-Resolution Dataset
### [Supplementary Material](https://github.com/widefield2sim/w2s/blob/master/w2s_supp.pdf)

> **Abstract:** *In fluorescence microscopy live-cell imaging, there is a critical trade-off between the signal-to-noise ratio and spatial resolution on one side, and the integrity of the biological sample on the other side. To obtain clean high-resolution (HR) images, one can either use microscopy techniques, such as structured-illumination microscopy (SIM), or apply denoising and super-resolution (SR) algorithms. However, the former option requires multiple shots that can damage the samples, and although efficient deep learning based algorithms exist for the latter option, no benchmark exists to evaluate these algorithms on the joint denoising and SR (JDSR) tasks.*
>
> *To study JDSR on microscopy data, we propose such a novel JDSR dataset, **W**idefield**2S**IM (W2S), acquired using a conventional fluorescence widefield and SIM imaging. W2S includes 144,000 real fluorescence microscopy images, resulting in a total of 360 sets of images. A set is comprised of noisy low-resolution (LR) widefield images with different noise levels, a noise-free LR image, and a corresponding high-quality HR SIM image. W2S allows us to benchmark the combinations of 6 denoising methods and 6 SR methods. We show that state-of-the-art SR networks perform very poorly on noisy inputs. Our evaluation also reveals that applying the best denoiser in terms of reconstruction error followed by the best SR method does not necessarily yield the best final result. Both quantitative and qualitative results show that SR networks are sensitive to noise and the sequential application of denoising and SR algorithms is sub-optimal. Lastly, we demonstrate that SR networks retrained end-to-end for JDSR outperform any combination of state-of-the-art deep denoising and SR networks*

## Widefield2SIM (W2S) Raw Data
![](https://github.com/widefield2sim/w2s/blob/master/figures/dataset.png)
We  obtain 5 types of LR images with different noise levels by taking a single raw image or averaging different numbers of raw images (of the same field of view). The more images we average (e.g., 2, 4, 8, and 16), the lower the noise level as shown in the figure. The noise-free LR images are the average of 400 raw images, and the HR images are obtained using structured-illumination microscopy (SIM). The multi-channel images are formed by mapping the three single-channel images of different wavelengths to RGB.

Raw data is available at: [https://datasets.epfl.ch/w2s/W2S_raw.zip](https://datasets.epfl.ch/w2s/W2S_raw.zip). 

To exactly reconstruct all averages from the raw data, we use the indices (where indices start at 0):

    index 249: used for "avg1", which is a single capture noisy image
    indices {0,1}: these images are averaged to obtain "avg2"
    indices {0,1,2,3}: these images are averaged to obtain "avg4"
    indices {0,1,2,3,4,5,6,7}: these images are averaged to obtain "avg8"
    indices {0,1, ..., 14, 15}: these images are averaged to obtain "avg16"

*The z-score is computed across all 360x400 captures; avg_value = 154.535390853, std_value = 66.02846351802853* 


## Models
### Pre-trained denoisers:
In folder ```net_data/trained_denoisers/```

### Pre-trained SRs:
In folder ```net_data/trained_srs/```

### Pre-trained JDSRs:
In folder ```net_data/trained_srs/```

### Reproducing results
To test the denoisers on W2S, run ```runtest.bash``` under code/denoising

To test the SR networks on W2S, run ```runtest.bash``` under code/SR

### Re-training of the networks:
Before training the networks run ```code/generate_h5f.ipynb``` to generate h5 files for training.

To train the denoisers on W2S, run ```runtrain.bash``` under code/denoising

To train the SR networks on W2S, run ```runtrain.bash``` under code/SR
