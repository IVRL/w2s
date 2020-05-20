# W2S: A Joint Denoising and Super-Resolution Dataset
### [Supplementary Material](https://github.com/widefield2sim/w2s/blob/master/w2s_supp.pdf)

> **Abstract:** *Denoising and super-resolution (SR) are fundamental tasks in imaging. These two restoration tasks are well covered in the literature, however, only separately. Given a noisy low-resolution (LR) input image, it is yet unclear what the best approach would be in order to obtain a noise-free high-resolution (HR) image. In order to study joint denoising and super-resolution (JDSR), a dataset containing pairs of noisy LR images and the corresponding HR images is fundamental. We propose such a novel JDSR dataset, **W**ieldfield**2S**IM (W2S), acquired using microscopy equipment and techniques. W2S is comprised of 144,000 real fluorescence microscopy images, used to form a total of 360 sets of images. A set is comprised of noisy LR images with different noise levels, a noise-free LR image, and a corresponding high-quality HR image. W2S allows us to benchmark the combinations of 6 denoising methods and 6 SR methods. We show that state-of-the-art SR networks perform very poorly on noisy inputs, with a loss reaching 14dB relative to noise-free inputs. Our evaluation also shows that applying the best denoiser in terms of reconstruction error followed by the best SR method does not yield the best result. The best denoising PSNR can, for instance, come at the expense of a loss in high frequencies, which is detrimental for SR methods. We lastly demonstrate that a light-weight SR network with a novel texture loss, trained specifically for JDSR, outperforms any combination of state-of-the-art deep denoising and SR networks.*

## Widefield2SIM (W2S) Dataset
![](https://github.com/widefield2sim/w2s/blob/master/figures/dataset.png)
We  obtain  5  LR  images  with different noise levels by taking a single raw image or averaging different numbers of raw  images  (of  the  same  field  of  view).  The  more  images  we  average  (e.g.,  2,  4,  8, and  16),  the  lower  the  noise  level  as  shown  in  the  figure.  The  noise-free  LR  imagesare the average of 400 raw images, and the HR images are obtained using structured-illumination microscopy (SIM). The multi-channel images are formed by mappingthe three single-channel images of different wavelengths to RGB.

To access the LR images with different noise levels of the training dataset

```cd data/train/avg{1,2,4,8,16}```

To access the clean LR images of the training dataset

```cd data/train/avg400```

To access the HR images of the training dataset

```cd data/train/sim```

To access the LR images with different noise levels of the test dataset

```cd data/test/avg{1,2,4,8,16}```

To access the clean LR images of the test dataset

```cd data/test/avg400```

To access the HR images of the test dataset

```cd data/test/sim```

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
