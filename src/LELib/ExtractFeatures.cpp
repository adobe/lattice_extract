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

#include "ExtractFeatures.h"

#include <iostream>

#include <opencv2/features2d.hpp>

#include "CommonParameters.h"

using namespace std;
using namespace cv;

bool PointCompare(const Point & p1, const Point & p2) {
    if (p1.x == p2.x) {
        return p1.y < p2.y;
    } else {
        return p1.x < p2.x;
    }
};

//if an MSER is included in another MSER, we discard it
//otherwise, we get many overlapping MSER features
//Not Used now- deprecated
void RemoveOverlaps(vector<vector<Point> > &regions) {
    int i = 0;
    while (i < regions.size()) {
        int j = i + 1;
        while (j < regions.size()) {
            if (includes(
                regions[i].begin(),
                regions[i].end(),
                regions[j].begin(),
                regions[j].end(),
                PointCompare)) {

                iter_swap(regions.begin() + j, regions.end() - 1);
                regions.pop_back();
            } else {
                j++;
            }
        }
        i++;
    }
}

//compute overlap based on intersection over union
void RemoveOverlaps(vector<Rect> &bboxes, vector<vector<Point> > &regions) {
    //construct a lambda to help sort the points and perform union
    auto pointSorter = [](const Point & p1, const Point & p2) {return (p1.x != p2.x) ? (p1.x < p2.x) : (p1.y < p2.y); };
    //pre-sort all the point sets
    for (auto it = regions.begin(); it != regions.end(); ++it) {
        sort(it->begin(), it->end(), pointSorter);
    }
    vector<Point> intersection;
    for (int i = 0; i < regions.size(); ++i) {
        for (int j = i + 1; j < regions.size(); ++j) {
            //check for bbox intersection to see if we need to test at all
            if ((bboxes[i] & bboxes[j]).area() > 0) {
                //get intersection of the point sets
                intersection.clear();
                set_intersection(regions[i].begin(), regions[i].end(), regions[j].begin(), regions[j].end(), back_inserter(intersection), pointSorter);

                //size of intersection
                size_t intMag = intersection.size();
                //size of union
                size_t unionMag = regions[i].size() + regions[j].size() - intersection.size();

                //compute intersection over union
                float iou = float(intMag) / float(unionMag);

                if (iou > FEATURE_OVERLAP_RATIO) {
                    //remove whichever region is smaller
                    if (regions[i].size() < regions[j].size()) {
                        swap(regions[i], regions[j]);
                        swap(bboxes[i], bboxes[j]);
                    }
                    swap(regions[j], regions.back());
                    swap(bboxes[j], bboxes.back());

                    regions.pop_back();
                    bboxes.pop_back();

                    --j;
                }
            }
        }
    }
}

std::vector<MSEREllipse> ExtractFeatures(const cv::Mat& image, int imgDistanceThreshold) {
    
    //auto mser = MSER::create();
    int minDim = image.cols;
    if(image.rows < minDim)
        minDim = image.rows;
    
    auto mser = MSER::create(5,imgDistanceThreshold,minDim*minDim,0.25,0.2,200,1,0.003,5);

    vector<vector<Point> > regions;
    vector<Rect> bboxes;
    vector<MSEREllipse> ellipses;

    Mat image_greyscale;
    cvtColor(image, image_greyscale, cv::COLOR_RGB2GRAY);

    //detect mser features
#ifdef VERBOSE
    cout << "Detecting MSER features..." << endl;
#endif
    mser->detectRegions(image_greyscale, regions, bboxes);
#ifdef VERBOSE
    cout << regions.size() << " MSER features detected!" << endl;
#endif

    //sort MSER features based on location
    for (vector<Point> & region : regions) {
        sort(region.begin(), region.end(), PointCompare);
    }

    //sort MSER features based on # of points involved
    //sort(regions.begin(), regions.end(),
    //    [](const vector<Point> & v1, const vector<Point> & v2) {
    //        return v1.size() > v2.size(); });


    RemoveOverlaps(bboxes, regions);
#ifdef VERBOSE
    cout << regions.size() << " after removing overlaps!" << endl;
#endif

    //create an ellipse for each MSER feature
    ellipses.reserve(regions.size());
    for (const auto& region : regions) {
        ellipses.push_back(MSEREllipse(region, image));
    }

    // remove ellipses that are bigger (both wider *and* taller) 
    // than half the image (as they cannot repeat)
    ellipses.erase(std::remove_if(ellipses.begin(), ellipses.end(), 
        [&image](const MSEREllipse& ellipse) -> bool {
            return ellipse.brect.size.width * 2 > image.cols
                && ellipse.brect.size.height * 2 > image.rows;
        }), ellipses.end());
    
    //remove ellipses that have 0 width or height
    ellipses.erase(std::remove_if(ellipses.begin(), ellipses.end(),
                                  [&image](const MSEREllipse& ellipse) -> bool {
                                      return ellipse.brect.size.width == 0
                                      || ellipse.brect.size.height == 0;
                                  }), ellipses.end());

    return ellipses;
}
