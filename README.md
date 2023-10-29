# Neural Field Convolutions By Repeated Differentiation
PyTorch  reference implementation

 Ntumba Elie Nsampi, Adarsh Djeacoumar, Hans-Peter Seidel, Tobias Ritschel, Thomas Leimkühler <br>
### [Project Page](https://neural-fields-conv.mpi-inf.mpg.de/) | [Paper](https://neural-fields-conv.mpi-inf.mpg.de/static/paper/compressed/neural_field_convolutions.pdf)

This repository contains the official authors' implementation associated with the paper 'Neural Field Convolutions by 
Repeated Differentiation,' 
accepted for publication in ACM Transactions on Graphics (Proceedings of SIGGRAPH Asia) in 2023.

Abstract: Neural fields are evolving towards a general-purpose continuous representation for visual computing.
Yet, despite their numerous appealing properties, they are hardly amenable to signal processing. 
As a remedy, we present a method to perform general continuous convolutions with general continuous signals such as neural fields.
Observing that piecewise polynomial kernels reduce to a sparse set of Dirac deltas after repeated differentiation, 
we leverage convolution identities and train a repeated integral field to efficiently execute large-scale convolutions. 
We demonstrate our approach on a variety of data modalities and spatially-varying kernels. 


## Quickstart
The commands below are used to set up a Conda environment, initiate the training process, and launch Tensorboard.
You can find the ground truths required for training our models [here](https://nextcloud.mpi-klsb.mpg.de/index.php/s/4kKWLKHT6k4g5Bk). 
Please ensure that you copy the downloaded files to their respective directories (data/GTs/geometry, data/GTs/images, data/GTs/videos).

(In case Jax fails to install or run, follow the instructions provided in the jax [repo](https://github.com/google/jax) to install your platform specific distribution)
```
conda create --name nfc python=3.9
conda activate nfc 
```

To install Pytorch, follow [these instructions](https://pytorch.org/get-started/locally/). 

```
conda install numpy imageio click
conda install -c conda-forge opencv matplotlib pillow cupy
conda install -c fastai opencv-python-headless
pip install decord functorch plyfile pysdf scikit_image scipy trimesh jax jaxlib tensorboard
pip install simpleimageio
```
```
cd scripts
sh train2d.sh
tensorboard --logdir=../logs --port=6006
```

## Pre-trained Models
You can download the pre-trained models [here](https://nextcloud.mpi-klsb.mpg.de/index.php/s/4kKWLKHT6k4g5Bk). Once downloaded, place the files in the './trained_models/...' folder. 
From the root directory, execute the following commands to run the pre-trained models and generate sample outputs.
(Please note that the models and desired output path should be specified in the 'eval.sh' files.)

```
cd scripts
sh eval.sh
```
eval.sh


```
cd "../"
pwd

python eval.py \
--model_path="./trained_models/images/model.pth" \   # model path
--kernel_path='./trained_models/kernels/gaussian' \  # kernel path if any
--save_path="./output/images" \                      # saving path
--modality=1 \                                       # which modality
--width=512 \                                        # width  
--height=512 \                                       # height
--depth=256 \                                        # depth
--block_size=32 \                                    # used in case of limited GPU memory 
--kernel_scale=20                                    # kernel scale (unit)
```

## Training using your own data
The first step in training with your own data is to generate the necessary ground truths. 
We offer sample data for this purpose, which you can find [here](https://nextcloud.mpi-klsb.mpg.de/index.php/s/4kKWLKHT6k4g5Bk) (Raw Data). 
The commands below serve as examples for generating 2D ground truths. 
The same principles apply to 3D data and videos.
```
cd scripts
sh generate_gt_2d.sh
```
To generate ground truths, users need to specify the parameters listed below.
```
# scripts/generate_gt_2d.sh
...
python mc_utils.py \
--path="" \           # path to the data
--sample_number=50 \  # Montecarlo number of samples
--save_path="" \      # path to save the GT
--half_size=0.0125 \  # minimal kernel size (see paper)
--order=0 \           # polynomial kernel order (see paper)
```
The generated ground truths can be loaded for visualization using the load_montecarlo_gt function in utils.py.
After generating the ground truths (GT), you can commence training by executing 
the commands below from the root directory. 
Please ensure that the paths to the ground truths are specified in the training scripts.

```
cd scripts
sh train2d.sh
```

[//]: # (## Kernel optimization)

[//]: # (To optimize your own kernel, please refer to 'kernel_optimization.py')

[//]: # (and follow the provided instructions.)

## FAQ

**Q1 : simpleimageio fails to install using pip:** 

If you are experiencing installation issues with simpleimageio, it is due to a broken cl.exe (MSVC). Possible workarounds:

* The [homepage](https://github.com/pgrit/SimpleImageIO) of simpleimageio is linked for reference:
    * Activate your environment before these steps
    * Clone, Build, Install, Cleanup:
    ```
    mkdir temp && cd temp
    git clone https://github.com/pgrit/SimpleImageIO.git
    cd SimpleImageIO
    mkdir build && cd build
    cmake .. -DCMAKE_BUILD_TYPE=Release
    cmake --build . --config Release
    cd ..
    pip install build 
    python -m build
    pip install ./dist/simpleimageio-*.whl
    cd .. && cd ..
    rm -rf temp
    ```
    That installs simpleimageio in your conda environment.

## Citation

if you find our work helpful to your research, consider citing us.

<section class="section" id="BibTeX">
  <div class="container is-max-desktop content">
    <pre><code>
 @article{Nsampi2023NeuralFC,
      author = {Ntumba Elie Nsampi and Adarsh Djeacoumar and Hans-Peter Seidel and Tobias Ritschel and Thomas Leimk{\"u}hler},
      title = {Neural Field Convolutions by Repeated Differentiation},
      year = {2023},
      issue_date = {December 2023},
      publisher = {Association for Computing Machinery},
      address = {Sydney, Australia},
      volume = {42},
      number = {6},
      url = {https://doi.org/10.1145/3618340},
      doi = {10.1145/3618340},
      journal = {ACM Trans. Graph.},
      month = {Dec},
      articleno = {206},
      numpages = {11},
}
</code></pre>
  </div>
</section>