# Pix2Pix
Implementation of Pix2Pix paper in PyTorch. I've tried to replicate the original paper as closely as possible, so if you read the paper the implementation should be identical. The results from this implementation I would say is on par with the paper, I'll include some examples results below.

## Results
The model was first trained on the Maps dataset also used in the Pix2Pix paper with the task converting satellite images to Google Maps like visualizations. The model was also trained on a fun anime dataset found on Kaggle and examples of the results are shown below. 


|1st row: Input / 2nd row: Generated / 3rd row: Target|
|:-:|
|<img src="results/results_anime.png" width="400" height="400" />|
|<img src="results/results_maps.png" width="450" height="320" />|

### Maps dataset


### Anime dataset
Input

## Installation

### Clone and install requirements
```bash
$ git clone https://github.com/SannaPersson/Pix2Pix.git
$ cd Pix2Pix
$ pip install requirements.txt
```

### Download pretrained weights on Maps dataset
Link to data:
Pretrained weights downloaded from this page: links coming soon


### Training
Edit the config.py file to match the setup you want to use. Then run train.py


## Pix2Pix paper
### Image-to-Image Translation with Conditional Adversarial Networks by Phillip Isola, Jun-Yan Zhu, Tinghui Zhou, Alexei A. Efros

#### Abstract
We investigate conditional adversarial networks as a general-purpose solution to image-to-image translation problems. These networks not only learn the mapping from input image to output image, but also learn a loss function to train this mapping. This makes it possible to apply the same generic approach to problems that traditionally would require very different loss formulations. We demonstrate that this approach is effective at synthesizing photos from label maps, reconstructing objects from edge maps, and colorizing images, among other tasks. Indeed, since the release of the pix2pix software associated with this paper, a large number of internet users (many of them artists) have posted their own experiments with our system, further demonstrating its wide applicability and ease of adoption without the need for parameter tweaking. As a community, we no longer hand-engineer our mapping functions, and this work suggests we can achieve reasonable results without hand-engineering our loss functions either.
```
@misc{isola2018imagetoimage,
      title={Image-to-Image Translation with Conditional Adversarial Networks}, 
      author={Phillip Isola and Jun-Yan Zhu and Tinghui Zhou and Alexei A. Efros},
      year={2018},
      eprint={1611.07004},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
