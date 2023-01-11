/*
Copyright 2022 Adobe. All rights reserved.
This file is licensed to you under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License. You may obtain a copy
of the License at http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under
the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR REPRESENTATIONS
OF ANY KIND, either express or implied. See the License for the specific language
governing permissions and limitations under the License.
*/

#ifndef COMMON_PARAMETERS_h
#define COMMON_PARAMETERS_h

#define USE_TBB

/******Utility defines**********/
#define NOMINMAX

/******GENERAL**********/
//each image will be rescaled such that its minimum dimension is less than or equal to MAX_IMG_DIM
#define MAX_IMG_DIM 512

/******FEATURE EXTRACTION**********/
//MSER features that have an intersection over union ratio bigger than FEATURE_OVERLAP_RATIO are overlapping.
//We will keep only one of the such overlapping features.
#define FEATURE_OVERLAP_RATIO 0.9
#define MIN_FEATURE_AREA 30

/******FEATURE CLUSTERING**********/
#define DISCARD_CLUSTERS_SMALLER_THAN 5

// This parameter has a huge impact in speed vs performance.
// higher -> less links will be made, and clustering will be faster, but we risk missing clusters,
// especially if the repetition count is low (e.g. a 2x2 grid). Ideally it should be about the
// same as the grid size, but we don't know that a priori. Set to 2 or more.
// Given two features a and b, we will look at the distance between them and only if this distances
// allows at least MIN_REPETITON repetitions in the image (e.g., dist_x*MIN_REPETITION < width), we will
// put an edge in the graph between the two features.
#define MIN_REPETITION 3

/******GRID EXTRACTION IN EACH FEATURE CLUSTER**********/
#define MIN_NUMBER_OF_FEATURE_INLIERS_OF_A_GRID 3
#define MAX_ALLOWED_PERCENTAGE_OF_GRID_ELEMENTS_MISSED 0.55

/*********FINAL GRID VERIFCATION***************/
//after a grid is detected in the input image, we will compute the average normalized cross correlation
//between its elements, if this is less than MIN_NORM_CROSS_CORR_SCORE, we will discard the grid
#define MIN_MATCHING_SCORE 0.8
#define MIN_TOTAL_FEATURES_GRID 10
#define MIN_GRID_AREA_RATIO 0.5


//debugging
//#define VERBOSE
//#define MY_DEBUG
//#define PROCESS_ALL

#if defined(WIN32) || defined(_WIN32)
#define PATH_SEPARATOR "\\"
#else
#define PATH_SEPARATOR "/"
#endif

// WARNING: DO NOT UNCOMMENT, unless you know what you are doing.
// save pattern images and grid image created from original Image.
// Save "No output" image names in noOut 
//#define DEBUG_BULK_PATTERN_EXTRACTION_RESULTS

//save the detected MSER features
//#define DEBUG_SAVE_FEATURES

//save each feature cluster
//#define DEBUG_SAVE_CLUSTERS

//save the grid that is generated from a cluster of features that has the maximum support of inliers
//#define DEBUG_SAVE_BIGGEST_LATTICE_IN_CLUSTER

//save sets of grids that have similar transformations
//#define DEBUG_SAVE_LATTICE_GROUPS

//After a set of lattices with similar transformations are identified, we assign their features into
//elements of a common grid. Save the result of this assignment.
//#define DEBUG_SAVE_GRID_ELEMENT_GROUPS

#endif /* COMMON_PARAMETERS_h */
