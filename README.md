# DTU 2024 Deep Learning: Using GAN's Discriminator to Classify Real and Fake Images

Collecting activations using `BigGAN-PyTorch`:

1. Download the weights: [BigGAN_ch96_bs256x8_138k.zip](https://drive.google.com/file/d/1nAle7FCVFZdix2--ks0r5JBkFnKw8ctW/view)
2. Unpack them under `BigGAN-PyTorch/weights/`
3. Copy file `src/discriminate.py` to `BigGAN-PyTorch/`, and run it. Make sure you've got some generated/real images in `generated-imageges`.
4. The file `activations.png` will be created in the current directory containing some conv layer activations.

Training a classifier using the last layer of the discriminator:

1. Download the dataset: [deepfake-and-real-images](https://www.kaggle.com/datasets/manjilkarki/deepfake-and-real-images), save it under `BigGAN-PyTorch/datasets/Dataset`
2. Copy file `src/train-disriminate-last-layer.py` to `BigGAN-PyTorch/`, and run it. Make sure you've got the dataset in `datasets/Dataset`!

### Generate Grad-CAM based heamap
1. copy `visualize.py` from `src` into `BigGAN-pytorch`   if its not present already in it.
2. navigate to `BigGAN-pytorch`
3. run using `python visualize.py`
