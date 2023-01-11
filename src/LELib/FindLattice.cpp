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
#include "FindLattice.h"

#include <opencv2/highgui/highgui.hpp>

#include "CommonParameters.h"

#ifdef USE_TBB
#include <tbb/tbb.h>
#endif

using namespace cv;

namespace {
    void sampleNumbers(size_t listSize, int sampleSize, std::vector<int> &samples) {
        std::vector<bool> alreadySampled;
        alreadySampled.resize(listSize, false);
        for (int i = 0; i<sampleSize; i++) {
            bool sampleFound = false;
            while (!sampleFound) {
                int seed = rand() % listSize;
                if (alreadySampled[seed])
                    continue;
                alreadySampled[seed] = true;
                samples.push_back(seed);
                sampleFound = true;
            }
        }
    }
}

//given a set of similar features, we will generate candidate horizontal and vertical translations
//each sampled transformation should be at least as big as imgDistanceThreshold
void findHorizontalAndVerticalTranslations(const std::vector<MSEREllipse> &ellipses,
                                           const std::vector<unsigned long> &cluster,
                                           std::vector<float> &horizontalTrans,
                                           std::vector<float> &verticalTrans,
                                           float imgDistanceThreshold)
{
    int noFeatures = cluster.size();
    
    //we will build a histogram of candidate horizantal and vertical transformations
    std::vector<std::vector<float> > horizontalTransCandidates;
    std::vector<std::vector<float> > verticalTransCandidates;
    
    for(int i=0; i<noFeatures; i++)
    {
        for(int j=i+1; j<noFeatures; j++)
        {
            Point2f diff;
            diff.x = (ellipses[cluster[j]].center.x - ellipses[cluster[i]].center.x);
            diff.y = (ellipses[cluster[j]].center.y - ellipses[cluster[i]].center.y);
            
            float horRep = fabs(diff.x);
            float verRep = fabs(diff.y);
            
            //cannot have both vertical and horizontal translation
            if(horRep > imgDistanceThreshold && verRep > imgDistanceThreshold)
                continue;
            
            //horizontal repetition
            if(horRep > imgDistanceThreshold)
            {
                //check if this horizontal transformation is similar to a previously found one
                //if not add it to the candidate transformation list
                bool found = false;
                for(int c=0; c<horizontalTransCandidates.size(); c++)
                {
                    if(fabs(horRep-horizontalTransCandidates[c][0]) < imgDistanceThreshold)
                    {
                        horizontalTransCandidates[c].push_back(horRep);
                        found = true;
                        break;
                    }
                }
                if(!found)
                {
                    horizontalTransCandidates.push_back(std::vector<float>());
                    horizontalTransCandidates[horizontalTransCandidates.size()-1].push_back(horRep);
                }
            }
            else if(verRep > imgDistanceThreshold)
            {
                bool found = false;
                for(int c=0; c<verticalTransCandidates.size(); c++)
                {
                    if(fabs(verRep-verticalTransCandidates[c][0]) < imgDistanceThreshold)
                    {
                        verticalTransCandidates[c].push_back(verRep);
                        found = true;
                        break;
                    }
                }
                if(!found)
                {
                    verticalTransCandidates.push_back(std::vector<float>());
                    verticalTransCandidates[verticalTransCandidates.size()-1].push_back(verRep);
                }
            }
        }
    }
    
    //for each bin in the histogram, we will use the median element
    for (int h=0; h<horizontalTransCandidates.size(); h++)
    {
        if(horizontalTransCandidates[h].size() == 0)
            continue;
        
        sort(horizontalTransCandidates[h].begin(), horizontalTransCandidates[h].end());
        horizontalTrans.push_back(horizontalTransCandidates[h][horizontalTransCandidates[h].size() / 2]);
        
        //alternative was to use the average, median seems to perform better
        //float horRep = 0;
        //for(int c=0; c<horizontalTransCandidates[h].size(); c++)
        //    horRep += horizontalTransCandidates[h][c];
        //horizontalTrans.push_back(horRep / horizontalTransCandidates[h].size());
    }
    
    for (int v=0; v<verticalTransCandidates.size(); v++)
    {
        if(verticalTransCandidates[v].size() == 0)
            continue;
        
        sort(verticalTransCandidates[v].begin(), verticalTransCandidates[v].end());
        verticalTrans.push_back(verticalTransCandidates[v][verticalTransCandidates[v].size() / 2]);
        
        //float verRep = 0;
        //for(int c=0; c<verticalTransCandidates[v].size(); c++)
        //    verRep += verticalTransCandidates[v][c];
        //verticalTrans.push_back(verRep / verticalTransCandidates[v].size());
    }
}

void findLatticesInClusterByTransformSampling(const Mat &img,
                                              std::vector<std::vector<unsigned long> > &clusters,
                                              std::vector<MSEREllipse> &ellipses,
                                              std::vector<Lattice> &extractedLattices,
                                              float imgDistanceThreshold) {
    int latticeIndex = 0;
    int numClusters = clusters.size();
	std::vector<Lattice> lattices;
	lattices.resize(numClusters);
	std::vector<int> latticeValid;
	latticeValid.resize(numClusters, 0);
    
#ifdef USE_TBB
    tbb::parallel_for((size_t)(0), (size_t)(numClusters), [&](size_t c) {
#else
        for (size_t c = 0; c < numClusters; c++) {
#endif
        
        //biggest inlier set
        std::vector<int> maxInliers;
        //the size of the grid in the biggest inlier set (#rows times #cols)
        int maxGridElementSize = -1;
        //the elements that correspond to each grid element in the biggest inlier set
        //for each (column,row) location, it will hold the feature indices that correspond to this location
        std::vector<std::vector<std::vector<int> > > maxGridElements;
        
        //printf("***cluster %d***\n", c);
        std::vector<float> horizontalTrans;
        std::vector<float> verticalTrans;
        
        findHorizontalAndVerticalTranslations(ellipses,
                                              clusters[c],
                                              horizontalTrans,
                                              verticalTrans,
                                              imgDistanceThreshold);
        
        //we also add a translation of 0 to account for vertical-only or horizontal-only repetition
        horizontalTrans.push_back(0.0f);
        verticalTrans.push_back(0.0f);
        
        //printf("candidate horizontal repetitions:\n");
        //for(int h=0; h<horizontalTrans.size(); h++)
        //{
        //    printf("%f ", horizontalTrans[h]);
        //}
        //printf("\n");
        
        //printf("candidate vertical repetitions:\n");
        //for(int v=0; v<verticalTrans.size(); v++)
        //{
        //    printf("%f ", verticalTrans[v]);
        //}
        //printf("\n");
        
        //evaluate the possible transformations
        for(int h=0; h<horizontalTrans.size(); h++)
        {
            for(int v=0; v<verticalTrans.size(); v++)
            {
                if(horizontalTrans[h] == 0 && verticalTrans[v] == 0)
                    continue;
                
                //printf("candidate transformation:%f %f\n", horizontalTrans[h], verticalTrans[v]);
                
                //For each candidate transformation, we will also evaluate
                //using each of the features in the cluster as a seed element.
                for(int r=0; r<clusters[c].size(); r++)
                {
                    //reference element
                    int seed = clusters[c][r];
                    int minXGridLocation = 0;
                    int minYGridLocation = 0;
                    int maxXGridLocation = 0;
                    int maxYGridLocation = 0;
            
                    std::vector<int> inliers;
                    std::vector<cv::Vec2i> gridLocations;
                    inliers.push_back(seed);
                    gridLocations.push_back(cv::Vec2i(0,0));
                    
                    for(int o=0; o<clusters[c].size(); o++)
                    {
                        if(o==r)
                            continue;
                
                        Point2f diff;
                        diff.x = (ellipses[clusters[c][o]].center.x - ellipses[seed].center.x);
                        diff.y = (ellipses[clusters[c][o]].center.y - ellipses[seed].center.y);
                
                        //vertical repetition only
                        if (horizontalTrans[h] == 0) {
                            if (fabs(diff.x) > imgDistanceThreshold)
                                continue;
                            
                            float verDiv = (diff.y) / verticalTrans[v];
                            float verDivFloor = floor(verDiv);
                            float verDivCeil = ceil(verDiv);
                            int yInd;
                            bool found = false;
                            
                            if (verDiv - verDivFloor < 0.05)
                            {
                                yInd = verDivFloor;
                                found = true;
                            }
                            else if(verDivCeil - verDiv < 0.05)
                            {
                                yInd = verDivCeil;
                                found = true;
                            }
                            
                            if(found)
                            {
                                inliers.push_back(clusters[c][o]);
                                gridLocations.push_back(cv::Vec2i(0, yInd));
                                if(yInd < minYGridLocation)
                                    minYGridLocation = yInd;
                                else if(yInd > maxYGridLocation)
                                    maxYGridLocation = yInd;
                            }
                        }
                        //horizontal repetition only
                        else if (verticalTrans[v] == 0) {
                            if (fabs(diff.y) > imgDistanceThreshold)
                                continue;
                            
                            float horDiv = (diff.x) / horizontalTrans[h];
                            float horDivFloor = floor(horDiv);
                            float horDivCeil = ceil(horDiv);
                            int xInd;
                            bool found = false;
                            if (horDiv - horDivFloor < 0.05)
                            {
                                xInd = horDivFloor;
                                found = true;
                            }
                            else if(horDivCeil - horDiv < 0.05)
                            {
                                xInd = horDivCeil;
                                found = true;
                            }
                            
                            if(found)
                            {
                                inliers.push_back(clusters[c][o]);
                                gridLocations.push_back(cv::Vec2i(xInd, 0));
                                if(xInd < minXGridLocation)
                                    minXGridLocation = xInd;
                                else if(xInd > maxXGridLocation)
                                    maxXGridLocation = xInd;
                            }
                        }
                        else
                        {
                            float horDiv = (diff.x) / horizontalTrans[h];
                            float verDiv = (diff.y) / verticalTrans[v];
                            float horDivFloor = floor(horDiv);
                            float horDivCeil = ceil(horDiv);
                            float verDivFloor = floor(verDiv);
                            float verDivCeil = ceil(verDiv);
                            
                            int yInd;
                            bool foundY = false;
                            
                            if (verDiv - verDivFloor < 0.05)
                            {
                                yInd = verDivFloor;
                                foundY = true;
                            }
                            else if(verDivCeil - verDiv < 0.05)
                            {
                                yInd = verDivCeil;
                                foundY = true;
                            }
                            
                            int xInd;
                            bool foundX = false;
                            if (horDiv - horDivFloor < 0.05)
                            {
                                xInd = horDivFloor;
                                foundX = true;
                            }
                            else if(horDivCeil - horDiv < 0.05)
                            {
                                xInd = horDivCeil;
                                foundX = true;
                            }
                            
                            if(foundY && foundX)
                            {
                                inliers.push_back(clusters[c][o]);
                                gridLocations.push_back(cv::Vec2i(xInd, yInd));
                                if(yInd < minYGridLocation)
                                    minYGridLocation = yInd;
                                else if(yInd > maxYGridLocation)
                                    maxYGridLocation = yInd;
                                
                                if(xInd < minXGridLocation)
                                    minXGridLocation = xInd;
                                else if(xInd > maxXGridLocation)
                                    maxXGridLocation = xInd;
                            }
                        }//if
                    } //o loop
                    
                    //element-grid cell linking
                    std::vector<std::vector<std::vector<int> > > foundGridLocations;
                    int gridXSize = maxXGridLocation-minXGridLocation+1;
                    int gridYSize = maxYGridLocation-minYGridLocation+1;
                    
                    //we will discard grids that have total number of elements smaller than a threshold
                    if(gridXSize*gridYSize < MIN_REPETITION)
                        continue;
                    
                    foundGridLocations.resize(gridXSize);
                    for(int e=0; e<gridXSize; e++)
                        foundGridLocations[e].resize(gridYSize);
                    for(int e=0; e<gridLocations.size(); e++)
                    {
                        foundGridLocations[gridLocations[e][0]-minXGridLocation][gridLocations[e][1]-minYGridLocation].push_back(inliers[e]);
                    }
                    
                    //how many grid cells are missed
                    int notFound = 0;
                    for(int e=0; e<foundGridLocations.size(); e++)
                    {
                        for(int f=0; f<foundGridLocations[e].size(); f++)
                        {
                            if(foundGridLocations[e][f].size()==0)
                                notFound += 1;
                        }
                    }
                    //if we missed more than a certain portion of the elements we discard this grid
                    if((float)notFound / (gridXSize*gridYSize) > MAX_ALLOWED_PERCENTAGE_OF_GRID_ELEMENTS_MISSED)
                        continue;
                    
                    //is this the grid which has the maximum number of detected elements so far?
                    if((gridXSize*gridYSize - notFound) > maxGridElementSize)
                    {
                        maxInliers.clear();
                        maxInliers.assign(inliers.begin(), inliers.end());
                        maxGridElements.assign(foundGridLocations.begin(), foundGridLocations.end());
                        maxGridElementSize = gridXSize*gridYSize - notFound;
                    }
                }//r loop
            }//vertical
        }//horizontal
        
        //if the biggest grid has sufficient feature inliers, we accept it
        Lattice l;
        if(maxInliers.size() > MIN_NUMBER_OF_FEATURE_INLIERS_OF_A_GRID)
        {
            l.latticeIndex = latticeIndex;
            latticeIndex += 1;
            l.inlierFeatureIndices.assign(maxInliers.begin(), maxInliers.end());
            
            //find the bounding box of the lattice
            cv::Vec2f minPt(img.cols, img.rows);
            cv::Vec2f maxPt(0, 0);
            
            for(int e=0; e<maxInliers.size(); e++)
            {
                if(ellipses[maxInliers[e]].center.x > maxPt(0))
                    maxPt(0) = ellipses[maxInliers[e]].center.x;
                if(ellipses[maxInliers[e]].center.x < minPt(0))
                    minPt(0) = ellipses[maxInliers[e]].center.x;
                
                if(ellipses[maxInliers[e]].center.y > maxPt(1))
                    maxPt(1) = ellipses[maxInliers[e]].center.y;
                if(ellipses[maxInliers[e]].center.y < minPt(1))
                    minPt(1) = ellipses[maxInliers[e]].center.y;
            }
            l.minPt = minPt;
            l.maxPt = maxPt;
            
            //refine the grid transformations based on the detected grid elements
            //we will initialize optTransformation based on the bounding box of the grid
            cv::Vec2f optTransformation(0,0);
            int countX=0;
            int countY=0;
            if(maxGridElements.size() > 1)
            {
                //we have more than 1 column
                optTransformation(0) = (maxPt(0)-minPt(0))/(maxGridElements.size()-1);
                countX += 1;
            }
            if(maxGridElements[0].size() > 1)
            {
                //we have more than 1 row
                optTransformation(1) = (maxPt(1)-minPt(1))/(maxGridElements[0].size()-1);
                countY += 1;
            }
            
            //look at pairwise neighbors to refine the grid transformations
            for(int e=0; e<maxGridElements.size(); e++)
            {
                for(int f=0; f<maxGridElements[e].size(); f++)
                {
                    if(maxGridElements[e][f].size() == 0)
                        continue;
                    
                    for(int e2=e; e2<maxGridElements.size(); e2++)
                    {
                        for(int f2=f; f2<maxGridElements[e].size(); f2++)
                        {
                            if(e2==e && f2==f)
                                continue;
                            
                            if(maxGridElements[e2][f2].size() == 0)
                                continue;
                            
                            if(e2>e)
                            {
                                optTransformation(0) = optTransformation(0) + (ellipses[maxGridElements[e2][f2][0]].center.x-ellipses[maxGridElements[e][f][0]].center.x);
                                countX += (e2-e);
                            }
                            if(f2>f)
                            {
                                optTransformation(1) = optTransformation(1) + (ellipses[maxGridElements[e2][f2][0]].center.y-ellipses[maxGridElements[e][f][0]].center.y);
                                countY += (f2-f);
                            }
                        }
                    }
                    
                }
            }
            
            if(countX > 0)
                optTransformation(0) = optTransformation(0) / countX;
            if(countY > 0)
                optTransformation(1) = optTransformation(1) / countY;
            
            l.firstGridTrans = cv::Vec2f(optTransformation(0), 0.0f);
            l.secondGridTrans = cv::Vec2f(0.0, optTransformation(1));
            
            l.gridElements.assign(maxGridElements.begin(), maxGridElements.end());
            lattices[c]=l;
			latticeValid[c] = 1;
            
#ifdef DEBUG_SAVE_BIGGEST_LATTICE_IN_CLUSTER
            //visualization
            Mat seedImg = img.clone();
            char buffer[1024];
            Vec3b colour(0, 0, 255);
            
            cv::Point2f refPt(-1,-1);
            cv::Vec2i refInd;
            for(int e=0; e<maxGridElements.size(); e++)
            {
                for(int f=0; f<maxGridElements[e].size(); f++)
                {
                    if(maxGridElements[e][f].size() == 0)
                        continue;
                    if(refPt.x == -1 && refPt.y == -1)
                    {
                        refPt = ellipses[maxGridElements[e][f][0]].center;
                        refInd(0) = e;
                        refInd(1) = f;
                    }
                    cv::ellipse(seedImg, ellipses[maxGridElements[e][f][0]].brect, colour, 2.0f);
                }
            }
            
            for(int e=0; e<maxGridElements.size()-1; e++)
            {
                for(int f=0; f<maxGridElements[e].size(); f++)
                {
                    cv::Point2f curPt;
                    if(maxGridElements[e][f].size() == 0)
                    {
                        curPt = refPt + cv::Point2f(optTransformation(0)*(e-refInd(0)),optTransformation(1)*(f-refInd(1)));
                    }
                    else
                        curPt = ellipses[maxGridElements[e][f][0]].center;
                    
                    cv::line(seedImg,
                             curPt,
                             curPt + cv::Point2f(optTransformation(0),0),
                             Vec3b(255,0,255), 2.0f);
                }
            }
            for(int e=0; e<maxGridElements.size(); e++)
            {
                for(int f=0; f<maxGridElements[e].size()-1; f++)
                {
                    cv::Point2f curPt;
                    if(maxGridElements[e][f].size() == 0)
                    {
                        curPt = refPt + cv::Point2f(optTransformation(0)*(e-refInd(0)),optTransformation(1)*(f-refInd(1)));
                    }
                    else
                        curPt = ellipses[maxGridElements[e][f][0]].center;
                        
                    cv::line(seedImg,
                             curPt,
                             curPt + cv::Point2f(0,optTransformation(1)),
                             Vec3b(255,0,255), 2.0f);
                }
            }
        
            sprintf(buffer, "./biggestCluster_%d.png", c);
            imwrite(buffer, seedImg);
#endif
        }
        
#ifdef USE_TBB
        });
#else
    }//cluster ***todo cluster1
#endif
		for (size_t i = 0; i < numClusters; ++i) {
			if (latticeValid[i]) {
				extractedLattices.push_back(lattices[i]);
			}
		}
}

//Given a detected lattice, this function aligns the horizontal
//and vertical separation lines of the lattice to the high gradient
//regions in the image.
//The input is the ElementCluster which is a grouping of feature elements
//to a commong grid that is formed by combining the different grids
//with similar transformations detected from different cluster of features
bool findLatticeOffset(const cv::Mat &img,
                       ElementCluster& ec)
{
    //todo if the transformation vectors are not horizontal/vertical
    //we would need to warp the image
    
    //find the gradient of the image in x and y directions
    cv::Mat tmp, dx, dy, gradX, gradY;
    cvtColor( img, tmp, COLOR_BGR2GRAY );
    cv::Sobel(tmp, dx, CV_16S, 1, 0, 3);
    cv::Sobel(tmp, dy, CV_16S, 0, 1, 3);
    cv::convertScaleAbs(dx, gradX);
    cv::convertScaleAbs(dy, gradY);
    
    //in order to define the range we are looking into,
    //we will choose a reference element and its immediate
    //horizontal and vertical neighbors
    //the separation lines will be somewhere in-between the
    //immediate grid cells
    int startX = 0;
    int endX = 0;
    int startY = 0;
    int endY = 0;
    int refCol = -1;
    int refRow = -1;
    
    //if it is vertical transformation only,
    //we set the x offset of the grid to be the
    //left-most starting point of one of the element bounding boxes
    if(cv::norm(ec.firstGridTrans) > 0){
        for(int c=0; c<ec.elements.size(); c++)
        {
            if(ec.elements[c].size() == 0)
                continue;
        
            int ref = c;
            int horNext = -1;
        
            for(int d=0; d<ec.elements.size(); d++)
            {
                if(ec.elements[d].size() == 0)
                    continue;
            
                if(horNext == -1)
                {
                    if(ec.elements[d][0].col - ec.elements[c][0].col == 1)
                    {
                        horNext = d;
                        break;
                    }
                }
            }
        
            if(horNext == -1)
                continue;
        
            refCol = ec.elements[ref][0].col;
        
            startX = ec.bboxCenterPts[ref](0);
            endX = ec.bboxCenterPts[horNext](0);
        
            //we found valid reference element with immediate horizontal neighbor
            //we are breaking the for loop
            break;
        }
    }
    
    if(cv::norm(ec.secondGridTrans) > 0)
    {
        for(int c=0; c<ec.elements.size(); c++)
        {
            if(ec.elements[c].size() == 0)
                continue;
        
            int ref = c;
            int verNext = -1;
        
            for(int d=0; d<ec.elements.size(); d++)
            {
                if(ec.elements[d].size() == 0)
                    continue;
            
                if(verNext == -1)
                {
                    if(ec.elements[d][0].row - ec.elements[c][0].row == 1)
                    {
                        verNext = d;
                        break;
                    }
                }
            }
        
            if(verNext == -1)
                continue;
        
            refRow = ec.elements[ref][0].row;
        
            startY = ec.bboxCenterPts[ref](1);
            endY = ec.bboxCenterPts[verNext](1);
        
            //we found valid reference element with immediate vertical neighbor
            //we are breaking the for loop
            break;
        }
    }
    
    //we could not find any valid neighbors, so this grid was not valid
    if(refRow < 0 && refCol < 0)
        return false;
    
    //we could not find any valid range
    if(startX == endX && startY == endY)
        return false;
    
    //vertical repetition only, we need to choose the horizontal repetition vector to be the grid
    //bounding box width
    if(cv::norm(ec.firstGridTrans) == 0)
    {
        bool assigned = false;
        for(int c=0; c<ec.elements.size(); c++)
        {
            if(ec.elements[c].size() == 0)
                continue;
            if(!assigned)
            {
                startX = ec.bboxMinPts[c](0);
                endX = ec.bboxMaxPts[c](0);
                assigned = true;
            }
            else
            {
                if(ec.bboxMinPts[c](0) < startX)
                    startX = ec.bboxMinPts[c](0);
                if(ec.bboxMaxPts[c][0] > endX)
                    endX = ec.bboxMaxPts[c](0);
            }
        }
        //update the horizontal repetition vector to be the same size as the bounding box width
        ec.firstGridTrans = cv::Vec2f(endX - startX,0);
        startX = endX;
        refCol = 0;
    }
    
    if(cv::norm(ec.secondGridTrans) == 0)
    {
        bool assigned = false;
        for(int c=0; c<ec.elements.size(); c++)
        {
            if(ec.elements[c].size() == 0)
                continue;
            if(!assigned)
            {
                startY = ec.bboxMinPts[c](1);
                endY = ec.bboxMaxPts[c](1);
                assigned = true;
            }
            else
            {
                if(ec.bboxMinPts[c](1) < startY)
                    startY = ec.bboxMinPts[c](1);
                if(ec.bboxMaxPts[c][1] > endY)
                    endY = ec.bboxMaxPts[c](1);
            }
        }
        //update the vertical repetition vector to be the same size as the bounding box height
        ec.secondGridTrans = cv::Vec2f(0, endY - startY);
        startY = endY;
        refRow = 0;
    }
    
    float maxScore = 0;
    cv::Vec2f offset(startX,startY);
    int grid_size_x = cv::norm(ec.firstGridTrans);
    int grid_size_y = cv::norm(ec.secondGridTrans);
    
    //there is no bounding box of features
    if(grid_size_x == 0 || grid_size_y == 0)
        return false;
    
    for(int x=startX; x<=endX; x++)
    {
        size_t score = 0;
        size_t count = 0;
        
        //go left
        /*for(int c=refCol; c>0; c--)
         {
         if(x - c*grid_size_x > 0)
         {
         for(int y=startY; y<=endY; y++)
         {
         score += gradX.at<unsigned char>(y, x - c*grid_size_x);
         count += 1;
         }
         }
         }*/
        
        for(int y=startY; y<=endY; y++)
        {
            score += gradX.at<unsigned char>(y, x);
            count += 1;
        }
        
        //go right
        /*for(int c=refCol+1; c<ec.nCols-1; c++)
         {
         if(x + c*grid_size_x < img.cols)
         {
         for(int y=startY; y<=endY; y++)
         {
         score += gradX.at<unsigned char>(y, x + c*grid_size_x);
         count += 1;
         }
         }
         }*/
        
        if((float)score/(float)count > maxScore)
        {
            maxScore = (float)score/(float)count;
            offset(0) = x;
        }
    }
    
    maxScore = 0;
    for(int y=startY; y<=endY; y++)
    {
        size_t score = 0;
        size_t count = 0;
        
        //go left
        /*for(int r=refRow; r>0; r--)
         {
         if(y - r*grid_size_y >= 0)
         {
         for(int x=startX; x<=endX; x++)
         {
         score += gradY.at<unsigned char>(y - r*grid_size_y, x);
         count += 1;
         }
         }
         }*/
        
        for(int x=startX; x<=endX; x++)
        {
            score += gradY.at<unsigned char>(y, x);
            count += 1;
        }
        
        //go right
        /*for(int r=refRow+1; r<ec.nRows-1; r++)
         {
         if(y + r*grid_size_y < img.rows)
         {
         for(int x=startX; x<=endX; x++)
         {
         score += gradY.at<unsigned char>(y + r*grid_size_y, x);
         count += 1;
         }
         }
         }*/
        
        if((float)score/(float)count > maxScore)
        {
            maxScore = (float)score/(float)count;
            offset(1) = y;
        }
    }
    
    ec.offset(0) = offset(0) - (refCol+1)*grid_size_x;
    ec.offset(1) = offset(1) - (refRow+1)*grid_size_y;
    
    return true;
}

/*********Below is our old method of detecting lattices by sampling features*/////////////

//ideally we should call the findLattices function for each cluster
void findLattices(const Mat &img,
                  Mat1d &distances,
                  std::vector<MSEREllipse> &ellipses,
                  double similarityThresh) {

    int ransacItrs = 100;
    int innerRansacItrs = 10;
    size_t noEllipses = ellipses.size();
    
    int imgWidth = img.cols;
    int imgHeight = img.rows;
    double imgDiagonalThreshold = sqrt(imgWidth*imgWidth + imgHeight * imgHeight) / 20.0;

    std::vector<int> maxInlierSeeds;
    std::vector<int> maxInliers;

    for (int i = 0; i<noEllipses; i++) {
        //choose a random feature
        //int seed1 = rand() % noEllipses;
        int seed1 = i;
        
        //find features that are similar to seed1
        //if running on a cluster of features, we don't need this
        std::vector<int> candidates;
        int nId = -1;
        
#ifdef MY_DEBUG
        Mat newImg = img.clone();
        Vec3b colour(0,0,255);
        cv::ellipse(newImg, ellipses[seed1].brect, colour);
#endif
        for (int j = 0; j<noEllipses; j++) {
            if (j == seed1)
                continue;

            if (distances(seed1, j) < similarityThresh)
            {
                candidates.push_back(j);
                
#ifdef MY_DEBUG
                Vec3b colour(255,0,0);
                cv::ellipse(newImg, ellipses[j].brect, colour);
#endif
            }
        }
        
#ifdef MY_DEBUG
        char buffer[10224];
        sprintf(buffer, "candidates%d.png", i);
        imwrite(buffer, newImg);
#endif

        if (candidates.size() < 5)
            continue;
        
        for (int itr = 0; itr<innerRansacItrs; itr++) {
#ifdef MY_DEBUG
            Mat newImg = img.clone();
            Vec3b colour(255,0,0);
            cv::ellipse(newImg, ellipses[seed1].brect, colour);
#endif
            //sample random features from the candidate list
            //for perspective images with homography, we need 3 additional samples
            //seed1: point 0,0
            //additionalSamples[0]: point 0,1
            //additionalSamples[1]: point 1,0
            //additionalSamples[2]: point 1,1
            std::vector<int> additionalSamples;
            sampleNumbers(candidates.size(), 2, additionalSamples);

#ifdef MY_DEBUG
            colour = Vec3b(0,0,255);
            cv::ellipse(newImg, ellipses[candidates[additionalSamples[0]]].brect, colour);
            cv::ellipse(newImg, ellipses[candidates[additionalSamples[1]]].brect, colour);
            char buf[1024];
            sprintf(buf, "./samples%d_%d.png", i,itr);
            imwrite(buf, newImg);
#endif
            
            //for perspective images with homography, we need to compute the homography matrix
            //map all other features to the fronto-parallel image with this homography before
            //we continue, this will make the image fronto-parallel

            //find the grid translations
            Point2f horizontalRep, verticalRep;
            horizontalRep.x = (ellipses[candidates[additionalSamples[0]]].center.x - ellipses[seed1].center.x);
            horizontalRep.y = (ellipses[candidates[additionalSamples[0]]].center.y - ellipses[seed1].center.y);

            verticalRep.x = (ellipses[candidates[additionalSamples[1]]].center.x - ellipses[seed1].center.x);
            verticalRep.y = (ellipses[candidates[additionalSamples[1]]].center.y - ellipses[seed1].center.y);

            //we want horizontal and vertical translations after the homography is applied
            //so the horizontal repetition should not have a significant y component
            //the vertical repetition should not have a significant x component
            //currently we use 5% of the image diagonal length as threshold
            if (fabs(horizontalRep.y) > imgDiagonalThreshold || fabs(verticalRep.x) > imgDiagonalThreshold)
                continue;

            //length of the horizontal and vertical repetition vectors
            float horizontalRepSize = sqrt(horizontalRep.x*horizontalRep.x + horizontalRep.y*horizontalRep.y);
            float verticalRepSize = sqrt(verticalRep.x*verticalRep.x + verticalRep.y*verticalRep.y);

            //maybe it is only a vertical repetition
            if (horizontalRepSize < imgDiagonalThreshold)
                horizontalRepSize = 0;

            //maybe it is only a horizontal repetition
            if (verticalRepSize < imgDiagonalThreshold)
                verticalRepSize = 0;

            //if both repetiton transformations are 0, continute
            if ((int)horizontalRepSize == 0 && (int)verticalRepSize == 0)
                continue;

            printf("Using seeds:\n");
            printf("%d ", seed1);
            for (int s = 0; s<additionalSamples.size(); s++) {
                printf("%d ", candidates[additionalSamples[s]]);
            }
            printf("\n");

            //normalize the horizontal and vertical repetition transformations
            if (horizontalRepSize != 0) {
                horizontalRep.x = horizontalRep.x / horizontalRepSize;
                horizontalRep.y = horizontalRep.y / horizontalRepSize;
            }
            if (verticalRepSize != 0) {
                verticalRep.x = verticalRep.x / verticalRepSize;
                verticalRep.y = verticalRep.y / verticalRepSize;
            }
            
            //try the original proposed transformation T along with T/2, T/3, T/4, T/5 in both directions
            std::vector<cv::Vec2f> candidateTransformations;
            for(int h=1; h<=5; h++)
            {
                if(horizontalRepSize == 0 && h>1)
                    continue;
                for(int v=1; v<=5; v++)
                {
                    if(verticalRepSize == 0 && v>1)
                        continue;
                    
                    candidateTransformations.push_back(cv::Vec2f(horizontalRepSize/h, verticalRepSize/v));
                }
            }

            printf("candidate transformation:\n");
            printf("hor:%f %f, ver:%f %f\n", horizontalRep.x, horizontalRep.y, verticalRep.x, verticalRep.y);

            for(int c=0; c<candidateTransformations.size(); c++)
            {
                std::vector<int> inliers;

                //find inliers
                for (int j = 0; j<noEllipses; j++) {
                    //if we operate with clusters, we don't need this check
                    if (distances(seed1, j) > similarityThresh)
                        continue;

                    //vector from the seed to the current feature
                    Point2f diff;
                    diff.x = (ellipses[j].center.x - ellipses[seed1].center.x);
                    diff.y = (ellipses[j].center.y - ellipses[seed1].center.y);

                    //if the horizontal and vertical differences are close to an integer repetition, we count it as inliers
                    //to the check, we want the remainder of the division to be less than a threshold (10% of the repetition vector now)
                    if ((int)(candidateTransformations[c][0]) == 0) {
                        //project this vector to the vertical repetition direction
                        float horLength = diff.x;
                        float verLength = verticalRep.dot(diff);

                        if (((int)verLength % (int)candidateTransformations[c][1] < candidateTransformations[c][1] / 10) ||
                            ((int)verLength % (int)candidateTransformations[c][1] > 9*candidateTransformations[c][1] / 10)) {
                            //it should be vertical repetition only
                            if (fabs(horLength) > imgDiagonalThreshold)
                                continue;

                            inliers.push_back(j);
                        }
                    } else if ((int)candidateTransformations[c][1] == 0) {
                        //project this vector to the horizontal repetition directions
                        float horLength = horizontalRep.dot(diff);
                        float verLength = diff.y;

                        //horLength or verLength can be negative, that's why we use absolute value
                        if (((int)fabs(horLength) % (int)candidateTransformations[c][0] < candidateTransformations[c][0] / 10) ||
                            ((int)fabs(horLength) % (int)candidateTransformations[c][0] > 9*candidateTransformations[c][0] / 10)) {
                            if (fabs(verLength) > imgDiagonalThreshold)
                                continue;

                            inliers.push_back(j);
                        }
                    } else {
                        //project this vector to the horizontal and vertical repetition directions
                        float horLength = horizontalRep.dot(diff);
                        float verLength = verticalRep.dot(diff);

                        if (((int)fabs(horLength) % (int)candidateTransformations[c][0] < candidateTransformations[c][0] / 10 ||
                             (int)fabs(horLength) % (int)candidateTransformations[c][0] > 9 * candidateTransformations[c][0] / 10) &&
                            ((int)fabs(verLength) % (int)candidateTransformations[c][1] < candidateTransformations[c][1] / 10 ||
                             (int)fabs(verLength) % (int)candidateTransformations[c][1] > 9*candidateTransformations[c][1] / 10)) {
                                inliers.push_back(j);
                            }
                    }
                }
                printf("%zd inliers found!\n", inliers.size());
                if (inliers.size() > maxInliers.size()) {
                    printf("max inliers at iteration %d\n", i);
                    printf("horizontalRep:%f %f %f\n", horizontalRep.x, horizontalRep.y, candidateTransformations[c][0]);
                    printf("verticalRep:%f %f %f\n", verticalRep.x, verticalRep.y, candidateTransformations[c][1]);
                    maxInliers.clear();
                    maxInliers.assign(inliers.begin(), inliers.end());

                    maxInlierSeeds.clear();
                    maxInlierSeeds.push_back(seed1);
                    for (int j = 0; j<additionalSamples.size(); j++)
                        maxInlierSeeds.push_back(candidates[additionalSamples[j]]);
                }
            }
        }
    }
    Mat seedImg = img.clone();
    for (int i = 0; i<maxInlierSeeds.size(); i++) {
        Vec3b colour(255, 0, 0);
        if (i == 0)
            colour = Vec3b(0, 255, 0);
        cv::ellipse(seedImg, ellipses[maxInlierSeeds[i]].brect, colour);
    }
    imwrite("./biggestClusterSeed.png", seedImg);
    
    Mat clusterImg = img.clone();
    for (int i = 0; i<maxInliers.size(); i++) {
        Vec3b colour(0, 0, 255);

        cv::ellipse(clusterImg, ellipses[maxInliers[i]].brect, colour);
    }
    imwrite("./biggestCluster.png", clusterImg);
}

void findLatticesInCluster(const Mat &img,
                           std::vector<std::vector<size_t> > &clusters,
                           Mat1d &distances,
                           std::vector<MSEREllipse> &ellipses,
                           double similarityThresh) {
    
    int noClusters = clusters.size();
    int ransacItrs = 100;
    int innerRansacItrs = 10;
    int imgWidth = img.cols;
    int imgHeight = img.rows;
    double imgDiagonalThreshold = sqrt(imgWidth*imgWidth + imgHeight * imgHeight) / 20.0;
    int noSamples = 2;
    
    for(int c=0; c<clusters.size(); c++)
    {
    
        size_t noEllipses = clusters[c].size();
        if(noEllipses < 5)
            continue;
    
        std::vector<int> maxInlierSeeds;
        std::vector<int> maxInliers;
    
        for (int i = 0; i<ransacItrs; i++) {
            //choose random features
            std::vector<int> samples;
            int seed1Ind = rand() % noEllipses;
            int seed1 = clusters[c][seed1Ind];
        
            for (int itr = 0; itr<innerRansacItrs; itr++) {

                //sample random features from the candidate list
                //for perspective images with homography, we need 3 additional samples
                //seed1: point 0,0
                //additionalSamples[0]: point 0,1
                //additionalSamples[1]: point 1,0
                //additionalSamples[2]: point 1,1
                std::vector<int> additionalSamples;
                for(int j=0; j<noSamples; j++)
                {
                    while(true)
                    {
                        int seed = rand() % noEllipses;
                        if(clusters[c][seed] == seed1)
                            continue;
                        if(find(additionalSamples.begin(), additionalSamples.end(), clusters[c][seed]) == additionalSamples.end())
                        {
                            additionalSamples.push_back(clusters[c][seed]);
                            break;
                        }
                    }
                }
            
                //for perspective images with homography, we need to compute the homography matrix
                //map all other features to the fronto-parallel image with this homography before
                //we continue, this will make the image fronto-parallel
            
                //find the grid translations
                Point2f horizontalRep, verticalRep;
                horizontalRep.x = (ellipses[additionalSamples[0]].center.x - ellipses[seed1].center.x);
                horizontalRep.y = (ellipses[additionalSamples[0]].center.y - ellipses[seed1].center.y);
            
                verticalRep.x = (ellipses[additionalSamples[1]].center.x - ellipses[seed1].center.x);
                verticalRep.y = (ellipses[additionalSamples[1]].center.y - ellipses[seed1].center.y);
            
                //we want horizontal and vertical translations after the homography is applied
                //so the horizontal repetition should not have a significant y component
                //the vertical repetition should not have a significat x component
                //currently we use 5% of the image diagonal length as threshold
                if (fabs(horizontalRep.y) > imgDiagonalThreshold || fabs(verticalRep.x) > imgDiagonalThreshold)
                    continue;
            
                //length of the horizontal and vertical repetition vectors
                float horizontalRepSize = sqrt(horizontalRep.x*horizontalRep.x + horizontalRep.y*horizontalRep.y);
                float verticalRepSize = sqrt(verticalRep.x*verticalRep.x + verticalRep.y*verticalRep.y);
            
                //maybe it is only a vertical repetition
                if (horizontalRepSize < imgDiagonalThreshold)
                    horizontalRepSize = 0;
            
                //maybe it is only a horizontal repetition
                if (verticalRepSize < imgDiagonalThreshold)
                    verticalRepSize = 0;
            
                //if both repetiton transformations are 0, continute
                if ((int)horizontalRepSize == 0 && (int)verticalRepSize == 0)
                    continue;
                
                //try the original proposed transformation T along with T/2, T/3, T/4, T/5 in both directions
                std::vector<cv::Vec2f> candidateTransformations;
                for(int h=1; h<=5; h++)
                {
                    if(horizontalRepSize == 0 && h>1)
                        continue;
                    for(int v=1; v<=5; v++)
                    {
                        if(verticalRepSize == 0 && v>1)
                            continue;
                        
                        candidateTransformations.push_back(cv::Vec2f(horizontalRepSize/h, verticalRepSize/v));
                    }
                }
                
                printf("Using seeds:\n");
                printf("%d ", seed1);
                for (int s = 0; s<additionalSamples.size(); s++) {
                    printf("%d ", additionalSamples[s]);
                }
                printf("\n");
                
                //normalize the horizontal and vertical repetition transformations
                if (horizontalRepSize != 0) {
                    horizontalRep.x = horizontalRep.x / horizontalRepSize;
                    horizontalRep.y = horizontalRep.y / horizontalRepSize;
                }
                if (verticalRepSize != 0) {
                    verticalRep.x = verticalRep.x / verticalRepSize;
                    verticalRep.y = verticalRep.y / verticalRepSize;
                }
                
                printf("candidate transformation:\n");
                printf("hor:%f %f, ver:%f %f\n", horizontalRep.x, horizontalRep.y, verticalRep.x, verticalRep.y);
                
                for(int ct=0; ct<candidateTransformations.size(); ct++)
                {
            
                    std::vector<int> inliers;
            
                    //find inliers
                    for (int j = 0; j<noEllipses; j++) {
                
                        //vector from the seed to the current feature
                        Point2f diff;
                        diff.x = (ellipses[clusters[c][j]].center.x - ellipses[seed1].center.x);
                        diff.y = (ellipses[clusters[c][j]].center.y - ellipses[seed1].center.y);
                
                        //if the horizontal and vertical differences are close to an integer repetition, we count it as inliers
                        //to the check, we want the remainder of the division to be less than a threshold (10% of the repetition vector now)
                        if ((int)(candidateTransformations[ct][0]) == 0) {
                            //project this vector to the vertical repetition direction
                            float horLength = diff.x;
                            float verLength = verticalRep.dot(diff);
                            
                            float verDiv = fabs(verLength) / candidateTransformations[ct][1];
                    
                            if (verDiv - floor(verDiv) < 0.05 || ceil(verDiv) - verDiv < 0.05) {
                                //it should be vertical repetition only
                                if (fabs(horLength) > imgDiagonalThreshold)
                                    continue;
                        
                                inliers.push_back(clusters[c][j]);
                            }
                        } else if ((int)candidateTransformations[ct][1] == 0) {
                            //project this vector to the horizontal repetition directions
                            float horLength = horizontalRep.dot(diff);
                            float verLength = diff.y;
                            
                            float horDiv = fabs(horLength) / candidateTransformations[ct][0];
                    
                            //horLength or verLength can be negative, that's why we use absolute value
                            if (horDiv - floor(horDiv) < 0.05 || ceil(horDiv) - horDiv < 0.05) {
                                if (fabs(verLength) > imgDiagonalThreshold)
                                    continue;
                        
                            inliers.push_back(clusters[c][j]);
                        }
                        } else {
                            //project this vector to the horizontal and vertical repetition directions
                            float horLength = horizontalRep.dot(diff);
                            float verLength = verticalRep.dot(diff);
                            
                            float horDiv = fabs(horLength) / candidateTransformations[ct][0];
                            float verDiv = fabs(verLength) / candidateTransformations[ct][1];
                    
                            if ((horDiv - floor(horDiv) < 0.05 || ceil(horDiv) - horDiv < 0.05) &&
                                (verDiv - floor(verDiv) < 0.05 || ceil(verDiv) - verDiv < 0.05)) {
                                    inliers.push_back(clusters[c][j]);
                                }
                        }
                    }
                    printf("%zd inliers found!\n", inliers.size());
                    if (inliers.size() > maxInliers.size()) {
                        printf("max inliers at iteration %d\n", i);
                        printf("horizontalRep:%f %f %f\n", horizontalRep.x, horizontalRep.y, horizontalRepSize);
                        printf("verticalRep:%f %f %f\n", verticalRep.x, verticalRep.y, verticalRepSize);
                        maxInliers.clear();
                        maxInliers.assign(inliers.begin(), inliers.end());
                
                        maxInlierSeeds.clear();
                        maxInlierSeeds.push_back(seed1);
                        for (int j = 0; j<additionalSamples.size(); j++)
                            maxInlierSeeds.push_back(additionalSamples[j]);
                    }
                }
            }
        }
#ifdef DEBUG_SAVE_BIGGEST_LATTICE_IN_CLUSTER
        if(maxInliers.size() > 3)
        {
            Mat seedImg = img.clone();
            for (int i = 0; i<maxInlierSeeds.size(); i++) {
                Vec3b colour(255, 0, 0);
                if (i == 0)
                    colour = Vec3b(0, 255, 0);
                cv::ellipse(seedImg, ellipses[maxInlierSeeds[i]].brect, colour);
            }
            char buffer[1024];
            sprintf(buffer, "./biggestClusterSeed_%d.png", c);
            imwrite(buffer, seedImg);
            for (int i = 0; i<maxInliers.size(); i++) {
                Vec3b colour(0, 0, 255);
        
                cv::ellipse(seedImg, ellipses[maxInliers[i]].brect, colour);
            }
            sprintf(buffer, "./biggestCluster_%d.png", c);
            imwrite(buffer, seedImg);
        }
#endif
    }
}
