# Lattice Extract

The goal of this project is to read a 2d image with a grid repetition, and output both the grid and the base element.

## Build
### Requirements
This project requires a modern C++ compiler that supports basic C++ 11 features.

Third party required libraries:

* Eigen3, included in the repository
* andres/graph, included in the repository
* OpenCV 3.3.1 (https://opencv.org/releases.html). Binaries are **not** included in the repository and you will have to download and build it.
* zlib should be built as part of OpenCV
* libpng should be built as part of OpenCV
* libtbb, this is a multihtreading library that should be installed and the corresponding include/library paths need to be added to the project settings or the Makefile


### Building on Windows
Open the Visual Studio solution `/proj/win/LatticeExtract/LatticeExtract.sln`

Currently, VS looks for OpenCV libraries in `/lib/opencv/lib/`. You must either copy your compiled OpenCV 3.3.1 library files here, or change the library directory to point to the location of OpenCV libraries.

Note: depending on your OpenCV installation, you may have to link against `opencv_core331.lib;opencv_features2d331.lib;opencv_highgui331.lib;opencv_imgcodecs331.lib;opencv_imgproc331.lib` or `opencv_world331.lib`.

### Building on Mac
Open the XCode project `/proj/mac/LatticeExtract/latticeExtraction.xcodeproj`

You may have to adjust the library directories to make sure XCode can find OpenCV libraries.

### Building with Makefile
We also provide a makefile to build the executable. Depending on where your OpenCV and libtbb libraries are located, you need to update the `OPENCV_LIB_PATH`, `TBB_LIB_PATH`, and `INCLUDEPATH` in the Makefile and then `make`. The executable `latticeExtractor` will be placed under the directory `build`.

## Running
The executable `TestApp` takes a single parameter, which is the (absolute or relative) path to the input image. If a grid is detected in the input image, two output images will be saved in the same directory as the executable. `<image_name>_finalGrid.png` will show the detected grid overlaid on the input image and `<image_name>_baseElement.png` will save out the repeating base element. Note that the saved output will be based on a resized input image (see the `Generic parameters` section below).

## Parameters
The algorithm has some parameters that can be (for now) adjusted in the `src/LELib/CommonParameters.h` file. Some additional parameters are embedded in the source code and we expect these parameters do not require tweaking.

### Generic parameters
* `MAX_IMG_DIR 512` indicates the maximum image size in pixels. Images whose minimum dimension is larger than this will be scaled down by half until they are within this size.
* `MY_DEBUG` or any define starting with `DEBUG_` activates saving intermediate steps to disk. May be useful for debugging.

### Feature Extraction
* `FEATURE_OVERLAP_RATIO 0.9` MSER features that have an intersection over union ratio bigger than this value are overlapping. We will keep only one of the such overlapping features.

* `const int size = 20` determine the size in pixels of the extracted feature.

### Feature Clustering
* `DISCARD_CLUSTERS_SMALLER_THAN 3` determines the minimum number of features of a cluster.

* `MIN_REPETITION 2` determines how many repetitions there can be in the image. This parameter has a big impact in speed vs performance. Higher -> less links will be made, and clustering will be faster, but we risk missing clusters, especially if the repetition count is low (e.g. a 2x2 grid). Ideally it should be about the same as the grid size, but we don't know that a priori. Set to 2 or more.

* `constexpr double app_min = 3.0` (in `GraphCutFeatures.cpp`) indicates what is a good match for feature appearance similarity. Reducing it too much will introduce false negatives. Increasing it too much will introduce false positives.
* `constexpr double amm_max = 5.0` (in `GraphCutFeatures.cpp`) indicates what is a bad match for feature appearance similarity. Reducing it too much will introduce false negatives. Increasing it too much will introduce false positives.
* There are a few more parameters in the file, documented in-line, but those shouldn't need tweaking.

### Grid Extraction for Each Cluster
* `MAX_ALLOWED_PERCENTAGE_OF_GRID_ELEMENTS_MISSED 0.65` We discard a candidate grid if the percentage of the missed grid elements is bigger than this threshold.

* `MAX_NUMBER_OF_FEATURE_INLIERS_OF_A_GRID 3` We will discard a candidate grid if the number of inlier features is less than this threshold.

### Final Verification of a grid
* `MIN_MATCHING_SCORE 0.8` After detecting a final grid, we compute a matching score between the grid elements and only if this matching score is above this value, we accept this grid. Note that while all previous analysis is done using image features, at this step we are comparing the actual image regions that correspond to the detected grid elements.

* `MIN_TOTAL_FEATURES_GRID 10` After detecting a final grid, we will also discard it if the total number of features that are supporting this grid is less than this value.

## TODO
* Currently, we are assuming the images are fronto-parallel. An additional homography estimation is necessary to rectify perspective images.

* Currently we are detecting only perfectly horizontal and vertical grids. We need to relax this assumption when sampling grid transformations for other type of grids.

* Current algorithm is fully automatic. In cases where user input is available, it will help to speed up and improve the algorithm.

* When clustering transformations, one transformation (e.g, t1, t2) might be able to generate another transformation (e.g., 2t1), shall we cluster these together? This might help with some of the problems above.

* After the final grid is detected, we can optionally see if it can be expanded to detect more rows/columns.
