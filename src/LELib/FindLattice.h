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

#pragma once

#include <vector>

#include <opencv2/core/core.hpp>

#include "MSEREllipse.h"

struct Lattice
{
    Lattice() : firstGridTrans(cv::Vec2f(0,0)), secondGridTrans(cv::Vec2f(0,0)), minPt(cv::Vec2f(0,0)), maxPt(cv::Vec2f(0,0)), latticeIndex(-1) {}
    
    std::vector<int> inlierFeatureIndices;
    cv::Vec2f firstGridTrans; //the first repeating transformation
    cv::Vec2f secondGridTrans; //the second repeating transformation
    cv::Vec2f minPt, maxPt; //bounding box of the lattice
    //for each <column,row> of the grid, it will hold the indices of the features
    //that correspond to this location
    std::vector<std::vector<std::vector<int> > > gridElements;
    int latticeIndex;
};

struct Node {
    int idx;        // index in 'ellipses' vector
    size_t lattice; // index in 'lattices' vector of the lattice they belong to
    size_t row;     // row in the lattice
    size_t col;     // column in the lattice
};

//for each lattice with similar transformation, we cluster the elements such that the elements in
//a given cluster belong to the same base element in the final repetition lattice
struct ElementCluster
{
    std::vector<std::vector<Node> > elements;
    //the position of each grid element in the image, these are clipped to be in image boundaries
    std::vector<std::vector<cv::Rect> > elementPosInImg;
    //the position of each grid element in the image, non-clipped
    std::vector<std::vector<cv::Rect> > elementPosInImg_nonClipped;
    //how well does the image region corresponding to a grid element matches with others
    std::vector<std::vector<float> > matchingScores;
    std::vector<cv::Vec2f> bboxCenterPts;
    std::vector<cv::Vec2f> bboxMinPts;
    std::vector<cv::Vec2f> bboxMaxPts;
    int nRows, nCols;
    cv::Vec2f firstGridTrans;
    cv::Vec2f secondGridTrans;
    cv::Vec2f offset;
};

//given a set of similar features, sample candidate horizontal and vertical translations
void findHorizontalAndVerticalTranslations(const std::vector<MSEREllipse> &ellipses,
                                           const std::vector<unsigned long> &cluster,
                                           std::vector<float> &horizontalTrans,
                                           std::vector<float> &verticalTrans,
                                           float imgDistanceThreshold);

void findLatticesInClusterByTransformSampling(const cv::Mat &img,
                                              std::vector<std::vector<unsigned long> > &clusters,
                                              std::vector<MSEREllipse> &ellipses,
                                              std::vector<Lattice> &extractedLattices,
                                              float imgDistanceThreshold);

//doesn't use cluster info, given the first sampled feature, finds features with small distance
void findLattices(const cv::Mat &img,
                  cv::Mat1d &distances,
                  std::vector<MSEREllipse> &ellipses,
                  double similarityThresh);

//uses the cluster info to sample features
void findLatticesInCluster(const cv::Mat &img,
                           std::vector<std::vector<unsigned long> > &cluster,
                           cv::Mat1d &distances,
                           std::vector<MSEREllipse> &ellipses,
                           double similarityThresh);

bool findLatticeOffset(const cv::Mat &img,
                       ElementCluster& ec);

