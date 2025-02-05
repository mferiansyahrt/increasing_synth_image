# increasing_synth_image

This project employs image processing approach as explained on [Day-to-Night Comparation github](https://github.com/mferiansyahrt/day_to_night_comparation).

## Super Resolution Method

The nighttime images were downscaled to 512 × 512 pixels and used as target data, while the low-resolution daytime images had a size of 256 × 256 pixels. After preparing the dataset, it was divided into training and testing sets. The model was then built and trained using the training set with an unpaired SISR approach and the proposed Direct Super-Resolution CycleGAN (DSR-CycleGAN) method. This process generated super-resolved synthetic nighttime images from low-resolution daytime inputs. The trained model was later tested using the testing set, with its performance evaluated both quantitatively and qualitatively.

<div align="center">
    <a href="./">
        <img src="./Figures/desain_peningkatan_resolusi_2_git.png" width="79%"/>
    </a>
</div>

The model was trained end-to-end without using a pre-trained model, ensuring it learns specific features directly from the dataset. Training was conducted on Google Colab Pro+ with an Nvidia A100 GPU. The model was developed using an unpaired SISR approach and DSR-CycleGAN with various generator architectures. Several hyperparameters were adjusted to support the research, including input and target image resolution, number of epochs, and learning rate.

<div align="center">
<table><thead>
  <tr>
    <th>Hyperparameter</th>
    <th></th>
  </tr></thead>
<tbody>
  <tr>
    <td>Dataset </td>
    <td>Dataset Gabungan</td>
  </tr>
  <tr>
    <td>Input Size</td>
    <td>256 × 256 piksel</td>
  </tr>
  <tr>
    <td>Target Size </td>
    <td>512 × 512 piksel</td>
  </tr>
  <tr>
    <td>Epoch</td>
    <td>100</td>
  </tr>
  <tr>
    <td>Learning Rate</td>
    <td>0,0002</td>
  </tr>
  <tr>
    <td>Training Image</td>
    <td>70 (Paired) + 100 (Unpaired) = 170 Images</td>
  </tr>
  <tr>
    <td>Testing Image</td>
    <td>50 Images</td>
  </tr>
</tbody>
</table>
</div>

In this stage, the image super-resolution model was developed using two frameworks: unpaired SISR and DSR-CycleGAN. The UNet-ViT generator from the UVCGAN model was found to deliver optimal performance from [Day-to-Night Comparation github](https://github.com/mferiansyahrt/day_to_night_comparation) project. Therefore, this study utilized the UNet-ViT generator for both frameworks to facilitate domain transfer from daytime to nighttime images

<div align="center">
<table><thead>
  <tr>
    <th>Framework</th>
    <th>Super-Resolution Model<br></th>
  </tr></thead>
<tbody>
  <tr>
    <td>Unpaired SISR</td>
    <td>UNet-ViT + ESRGAN+</td>
  </tr>
  <tr>
    <td rowspan="3">DSR-CycleGAN</td>
    <td>UNet-ViT + Upsampling</td>
  </tr>
  <tr>
    <td>UNet-ViT + ESRGAN+</td>
  </tr>
  <tr>
    <td>UNet-ViT + Enlighten-GAN</td>
  </tr>
</tbody>
</table>
</div>

Various image super-resolution models were explored, each paired with a corresponding upscaling model. As previously explained, the UNet-ViT generator was used for domain transfer, while the upscaling model handled spatial resolution enhancement.

### Unpaired Single Image Super-Resolution (SISR)

Traditional Single-Image Super Resolution (SISR) methods typically rely on paired datasets, where high-resolution (HR) images are downscaled to create corresponding low-resolution (LR) versions. In this study, we utilize both paired and unpaired datasets, employing an unpaired SISR framework based on a semi-supervised Generative Adversarial Network (GAN) approach. Unlike conventional unpaired SISR methods that operate within the same domain—such as enhancing the resolution of nighttime images by mapping from low to high resolution within the nighttime domain—our research focuses on mapping daytime images to the nighttime domain while simultaneously increasing their resolution.

<div align="center">
    <a href="./">
        <img src="./Figures/new_arsitektur_unpairedSISR_git.png" width="74%"/>
    </a>
</div>

A common issue with paired SISR methods is that the downscaling process, often performed using mathematical operations like bicubic interpolation, may not effectively generate super-resolved images in unpaired data scenarios. Therefore, this study adopts an unpaired SISR approach to develop a Super-Resolution (SR) model, particularly considering the unpaired nature of the combined dataset, which includes the BDD dataset. In this framework, we employ an L1 loss function for Lrec.

### Direct Super-Resolution CycleGAN (DSR-CycleGAN)
The proposed Direct Super-Resolution CycleGAN (DSR-CycleGAN) shares a similar overall structure with the standard CycleGAN architecture. However, in this study, the DSR-CycleGAN introduces specific modifications. While some research employs discriminators for both high-resolution (HR) and low-resolution (LR) images, and others use LR discriminators within the same domain, this study applies the HR discriminator exclusively to different domains.

<div align="center">
    <a href="./">
        <img src="./Figures/dsrcgan_revisi_git.png" width="74%"/>
    </a>
</div>

In this architecture, SRDay and SRNight consist of a UNet-ViT generator paired with various upsampling models. The UNet-ViT generator utilizes 12 transformer blocks and employs an L1 loss function. The SRNight model processes LR daytime images with a resolution of 256 × 256 pixels, outputting synthetic SR nighttime images at 512 × 512 pixels; conversely, the SRDay model performs the reverse operation. Both DDay and DNight discriminators use a 70 × 70 PatchGAN architecture with L2 loss, receiving HR images as real labels and SR images as fake labels. Downscaling methods such as bicubic interpolation and convolutional downsampling are mathematical operations applied sequentially. Bicubic interpolation is chosen for producing smoother-textured LR images, while convolutional downsampling is employed to reduce noise artifacts, especially when the framework is trained with perceptual loss

## Downstream Task: Object Detection
