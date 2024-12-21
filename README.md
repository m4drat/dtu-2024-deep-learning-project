# DTU 2024 Deep Learning: Using GAN's Discriminator to Classify Real and Fake Images

### Collecting activations using `BigGAN-PyTorch`:

1. Download the weights: [BigGAN_ch96_bs256x8_138k.zip](https://drive.google.com/file/d/1nAle7FCVFZdix2--ks0r5JBkFnKw8ctW/view)
2. Unpack them under `BigGAN-PyTorch/weights/`
3. Copy file `src/discriminate.py` to `BigGAN-PyTorch/`, and run it. Make sure you've got some generated/real images in `generated-imageges`.
4. The file `activations.png` will be created in the current directory containing some conv layer activations.

### Training a classifier using the last layer of the discriminator:

1. Download the dataset: [deepfake-and-real-images](https://www.kaggle.com/datasets/manjilkarki/deepfake-and-real-images), save it under `BigGAN-PyTorch/datasets/Dataset`
2. Copy file `src/train-disriminate-last-layer.py` to `BigGAN-PyTorch/`, and run it. Make sure you've got the dataset in `datasets/Dataset`!

### Generate Grad-CAM based heamap
1. Copy `visualize.py` from `src` into `BigGAN-pytorch`   if its not present already in it.
2. Add path to your input image in the script
3. Navigate to `BigGAN-pytorch`
4. Run using `python visualize.py`

### CNN part
There are two files with weights `faces_weights_128.pth` and `faces-weights-gan.pth` which correspond to experiments 1 and 2 respectfully.
For the experiments just run the corresponding notebooks (experiment1 and experiment2)

If you want to train the models yourself go to the cnn/training folder and run an appropriate script. There are also bash scripts for making jobs on HPC
