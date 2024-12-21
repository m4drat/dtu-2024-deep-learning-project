# DTU 2024 Deep Learning: Using GAN's Discriminator to Classify Real and Fake Images

### Collecting activations using `BigGAN-PyTorch`:

1. Download the weights: [BigGAN_ch96_bs256x8_138k.zip](https://drive.google.com/file/d/1nAle7FCVFZdix2--ks0r5JBkFnKw8ctW/view)
2. Unpack them under `BigGAN-PyTorch/weights/`
3. Copy file `src/discriminate.py` to `BigGAN-PyTorch/`, and run it. Make sure you've got some generated/real images in `generated-imageges`.
4. The file `activations.png` will be created in the current directory containing some conv layer activations.

### Training a classifier using the last layer of the discriminator:

1. Download the dataset: [deepfake-and-real-images](https://www.kaggle.com/datasets/manjilkarki/deepfake-and-real-images), save it under `BigGAN-PyTorch/datasets/Dataset`
2. Copy file `src/train-disriminate-last-layer.py` to `BigGAN-PyTorch/`, and run it. Make sure you've got the dataset in `datasets/Dataset`
3. Or just download pre-trained model from here: [best_model_params1733195785.9681547-16-0.8005](https://drive.google.com/file/d/1DLahtZ8jHl7xZvRTFduu-ocUjeiVtXY1/view?usp=sharing) and use it with `gan-discriminator-experiments.py`

### How to run visualizer of Grad-CAM heatmap
1. Clone the repo using the `--recurse-submodules` , this tag clones the dependency submodule which we have simlined in this repo.
2. you have to fetch 2 models ,Get the `model_weights` file from the google drive here . Download the model weights file from google file [weights_file](https://drive.google.com/drive/folders/1IHT0uvJRCyn7BZ3hq8qoNGF1VYCLCRI9?usp=sharing)  and also  and [D.pth](https://drive.google.com/drive/folders/1IHT0uvJRCyn7BZ3hq8qoNGF1VYCLCRI9) paste it under `src/visualize/model_weights` 
3. Append your python path with the path to the `BigGAN-PyTorch` submodule so that all modules within the library are visible to the python environment. Alternatively you can use the following `setup.py` file to build your wheel and install using pip, this will change references and hence is not recommended.
```
setup(
    name='BigGAN',
    version='0.1',
    packages=find_packages(),  # Automatically finds your packages
)
```
4. Run `pip install -r requirements.txt` to install all the dependencies.
5. cd to `src/visualize` and run `python visualize.py` or run the .ipynb notebook cells, the heatmap generated images can be found under `src/visualize/outputs` 
### Alternate method to generate Grad-CAM based heatmap
1. Copy `visualize.py` from `src` into `BigGAN-PyTorch`   if its not present already in it.
2. We assume that `weights/138k` folder is also present in `BigGAN-PyTorch`
2. Add path to your input image in the script and path to transfer learned model
3. Navigate to `BigGAN-pytorch`
4. Run using `python visualize.py`

### CNN part
There are two files with weights `faces_weights_128.pth` and `faces-weights-gan.pth` which correspond to experiments 1 and 2 respectfully.
For the experiments just run the corresponding notebooks (experiment1 and experiment2)

If you want to train the models yourself go to the cnn/training folder and run an appropriate script. There are also bash scripts for making jobs on HPC
