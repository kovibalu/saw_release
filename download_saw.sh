#!/bin/bash

# Downloads the full Shading Annotations in the Wild (SAW) dataset
# Files:
# Crowd-sourced annotations: saw_annotations_json.zip -> ~200 MB
# Photo ID list for trainining/validation/test set: saw_splits.zip -> ~20 KB
# Pixel labels for (NS-ND, NS-SB, S classes, see paper for details): saw_pixel_labels.zip -> ~55 MB
# Pixel labels visualization (NS-ND: red, NS-SB: cyan, S: green, see paper for details): saw_label_images_512.zip -> ~1.8 GB
# Resized images: saw_images_512.zip -> ~1.8 GB
# NYUv2 depth, normals and valid pixel masks: saw_nyu-depthv2.zip -> ~4.0 GB
# Our trained net weights: saw_pixelnet-614.zip -> ~0.7 GB
# Decompositions for all photos for each baseline: [algorithm slug]-[algorithm ID].zip (e.g. bell2014_densecrf-1141.zip)-> ~2.5 GB for each algorithm

function download_unzip {
	if [ ! -f "$1/${2}.zip" ]; then
		echo "Downloading ${2}..."
		wget -O "$1/${2}.zip" https://s3.amazonaws.com/labelmaterial/release/cvpr2017_saw/${2}.zip
	else
		echo "Already downloaded $2"
	fi
	if [ ! -d "$1/$2" ]; then
		echo "Unzipping ${2}..."
		unzip -o -d "$1/$2" "$1/${2}.zip"
	else
		echo "Already unzipped $2"
	fi
}

PARENT_DIR="saw"
if [ ! -d "$PARENT_DIR" ]; then
	mkdir "$PARENT_DIR"
fi
FILES=("saw_annotations_json" "saw_splits" "saw_pixel_labels" "saw_images_512" "saw_label_images_512" "saw_nyu-depthv2" "saw_pixelnet-614")
for file in "${FILES[@]}"
do
	download_unzip $PARENT_DIR $file
done

DECOMP_DIR="saw/saw_decomps"
if [ ! -d "$DECOMP_DIR" ]; then
	mkdir "$DECOMP_DIR"
fi
ALGOS=("baseline_reflectance-759" "bell2014_densecrf-1141" "garces2012_clustering-1221" "grosse2009_color_retinex-633" "shen2011_optimization-426" "zhao2012_nonlocal-709" "zhou2015_reflprior-1281")
for algo in "${ALGOS[@]}"
do
	download_unzip "$DECOMP_DIR" "$algo"
done

