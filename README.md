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


In this stage, the image super-resolution model was developed using two frameworks: unpaired SISR and DSR-CycleGAN. The UNet-ViT generator from the UVCGAN model was found to deliver optimal performance. Therefore, this study utilized the UNet-ViT generator for both frameworks to facilitate domain transfer from daytime to nighttime images

Various image super-resolution models were explored, each paired with a corresponding upscaling model. As previously explained, the UNet-ViT generator was used for domain transfer, while the upscaling model handled spatial resolution enhancement.

## Downstream Task: Object Detection
