# Shading Annotations in the Wild (SAW)

Code and data for paper "[Shading Annotations in the Wild](http://opensurfaces.cs.cornell.edu/publications/saw/)".

## Installation
### Dependencies
Our code was tested on Ubuntu 14.04. As a first step, clone our repo:
```bash
git clone https://github.com/kovibalu/saw_release.git
```

Then install the python dependencies by running:
```bash
sudo ./install/install_python.sh
```

If you would like to run our trained model, you will need to install [Caffe](http://caffe.berkeleyvision.org). We slightly modified the implementation of [Bansal et. al](https://github.com/aayushbansal/MarrRevisited) for our purposes. To check out our Caffe version which is included as a submodule, run:
```bash
git submodule update --init --recursive
```

Then build Caffe after editing the ``Makefile.config`` depending on your configuration with:
```bash
cd caffe
make all -j
make pycaffe -j
```

### Download Data
To download all data related to the dataset, run:
```bash
./download_saw.sh
```

The whole dataset download size is ~28.0GB, please see the documentation in the
script for a detailed breakdown of sizes for the different parts of the
dataset. For detailed documentation on the format of the downloaded annotations
in `saw/saw_annotations_json` see `ANNO_FORMAT.md`.

## Usage
### Precision-recall Curves
To generate the precision-recall curves in our paper for all baselines and our method, run:
```bash
python main.py generate_pr
```
You can select which baselines to evaluate in ``main.py``.

### Generating Pixel Labels
To generate the pixel labels from the SAW annotations and [NYUv2 depth dataset](http://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html) depth and normal maps, run:
```bash
python main.py generate_labels
```

## Citation
Please cite our paper if you use our code or data:
```
@article{kovacs17shading,
	author = "Balazs Kovacs and Sean Bell and Noah Snavely and Kavita Bala",
	title = "Shading Annotations in the Wild",
	journal = "Computer Vision and Pattern Recognition (CVPR)",
	year = "2017",
}
```

## Contact
Please contact [Balazs Kovacs](http://bkovacs.com) with any questions.
