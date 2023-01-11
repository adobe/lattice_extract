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

#include "DetectLatticeInImage.h"

#include <array>
#include <iostream>
#include<vector>
#include <opencv2/highgui/highgui.hpp>

#include "../LELib/GraphCutFeatures.h"
#include "../LELib/GraphCutGridElements.h"
#include "../LELib/Utils.h"
#include "../LELib/FindLattice.h"
#include "../LELib/MeanShiftFeatures.h"
#include "../LELib/ExtractFeatures.h"
#include "../LELib/MSEREllipse.h"
#include "../LELib/CommonParameters.h"


using namespace std;
using namespace cv;

#ifdef DEBUG_BULK_PATTERN_EXTRACTION_RESULTS
//bool detectLatticeInImage(cv::Mat& image, string imageName, string baseOutputDir) {
bool detectLatticeInImage(cv::Mat& image, string imageName, cv::Mat &basePattern, string baseOutputDir, vector<string> &noOut, int& outMaxIterCount)
#else
bool detectLatticeInImage(cv::Mat& image, cv::Mat &basePattern, int& outMaxIterCount) 
#endif
{
    if (!image.data) return false;

    //uniformly scale the input image such that we don't operate at a resolution bigger than MAX_IMG_DIM
    int minImgDim = image.rows;
    if (image.cols < minImgDim)
        minImgDim = image.cols;

    double scale = 1.0;

    if (minImgDim > MAX_IMG_DIM) {
        //scale = scale/2.0;
        //minImgDim = minImgDim/2;
        scale = float(MAX_IMG_DIM) / float(minImgDim);
    }

    {
        Mat tmp;
        resize(image, tmp, Size(), scale, scale, cv::INTER_AREA);
        image = tmp;
    }

    //define some thresholds
    //Any grid transformation that is smaller than imgDistanceThreshold will be ignored.
    //Any MSER feature that has area smaller than imgDistanceThreshold will be ignored.
    float imgDistanceThreshold = std::min(image.rows, image.cols) / 20.0f;

    //extract MSER features
    vector<MSEREllipse> ellipses = ExtractFeatures(image, MIN_FEATURE_AREA);

#ifdef VERBOSE
    printf("Found %d features\n", ellipses.size());
#endif

    //if we extract too few feature we return
    if (ellipses.size() < DISCARD_CLUSTERS_SMALLER_THAN)
        return false;

#ifdef DEBUG_SAVE_FEATURES
    {
        int ctr = 1;
        Mat featureImg = image.clone();
        for (MSEREllipse ellipse : ellipses) {
            RotatedRect rect = ellipse.brect;
            Vec3b color = Int2Color(ctr);
            cv::ellipse(featureImg, ellipse.brect, color);
            ++ctr;
        }

        char buffer[1024];
        sprintf(buffer, "%s%s%s_features.png", baseOutputDir.c_str(), PATH_SEPARATOR, imageName.c_str());
        imwrite(buffer, featureImg);
    }
#endif

    //cluster MSER features based on feature similarity
    //std::vector<std::vector<unsigned long>> clusters = GraphCutFeatures(ellipses, image);

    //std::vector<std::vector<unsigned long>> clusters = MeanShiftFeatures(ellipses, image);

    //first cluster based on shape similarity
    float shapeWeight = 1.0f;
    float appearanceWeight = 0.0f;
    bool treeSearch = false; //shape similarity is not metric
	int maxMaxIterCount = -1;
	int maxIterCount = -1;
    std::vector<std::vector<unsigned long>> shapeClusters = MeanShiftFeatures(maxIterCount, ellipses, image, shapeWeight, appearanceWeight, treeSearch);
	if (maxIterCount > maxMaxIterCount)
		maxMaxIterCount = maxIterCount;

    //cluster each group further based on appearance similarity
    std::vector<std::vector<unsigned long> > clusters;
    appearanceWeight = 1.0;
    shapeWeight = 1.0;
    treeSearch = true;
    for (int c = 0; c < shapeClusters.size(); c++) {
        //current ellipses to cluster
        std::vector<MSEREllipse> curEllipses;
        curEllipses.reserve(shapeClusters[c].size());
        for (int e = 0; e < shapeClusters[c].size(); e++)
            curEllipses.push_back(ellipses[shapeClusters[c][e]]);

        //perform clustering
        std::vector<std::vector<unsigned long>> finalClusters = MeanShiftFeatures(maxIterCount, curEllipses, image, shapeWeight, appearanceWeight, treeSearch);
		if (maxIterCount > maxMaxIterCount)
			maxMaxIterCount = maxIterCount;

        //printf("Shape cluster %d results in %d clusters!\n", c, finalClusters.size());

        //convert indices to be compatible with the original ellipse list
        for (int f = 0; f < finalClusters.size(); f++) {
            std::vector<unsigned long> newCluster;
            for (int e = 0; e < finalClusters[f].size(); e++)
                newCluster.push_back(shapeClusters[c][finalClusters[f][e]]);
            clusters.push_back(newCluster);
        }

    }

	outMaxIterCount = maxMaxIterCount;

    // sort clusters by size
    //sort(clusters.begin(), clusters.end(),
    //     [](const vector<unsigned long>& c1, const vector<unsigned long>& c2) {
    //         return c1.size() > c2.size();
    //     });

#ifdef VERBOSE
    printf("There are %zu clusters in the final partitions:\n", clusters.size());
#endif

#ifdef DEBUG_SAVE_CLUSTERS
    for (size_t i = 0; i < clusters.size(); ++i) {
        printf("cluster %i has %zd elements\n", i, clusters[i].size());

        Mat clusterImg = image.clone();
        for (int j = 0; j < clusters[i].size(); j++) {
            const MSEREllipse& e = ellipses[clusters[i][j]];
            RotatedRect rect = e.brect;
            Vec3b color(0, 0, 255); // BGR
            cv::ellipse(clusterImg, e.brect, color);
        }

        char buffer[1024];
        //sprintf(buffer, "%s%s%s%s_cluster%d.png", ".\\",imageName.c_str(), PATH_SEPARATOR, "clusters\\", i);
        sprintf(buffer, "cluster%03d.png", i);
        printf("Writing %s: %d\n", buffer,
            imwrite(buffer, clusterImg));
    }

#endif

    //find candidate grids, i.e., lattices, in each cluster of similar MSER features
    //for each cluster, we will return only one lattice that has the max support
    std::vector<Lattice> lattices;
    findLatticesInClusterByTransformSampling(image, clusters, ellipses, lattices, imgDistanceThreshold);

    //separate the lattices detected for each cluster into bins based on the transformation vectors
    const float epsilon = imgDistanceThreshold / 2;
    typedef std::array<cv::Vec2f, 2> Key;
    std::vector<std::vector<Lattice>> lattice_groups = SeparateInBins<Key, Lattice>(lattices,
        [](const Lattice& lattice) -> Key {
            return { lattice.firstGridTrans, lattice.secondGridTrans };
        },
        [epsilon](const Key& k1, const Key& k2) -> bool {
            //todo::assume for now that transformations have the same orientation, i.e., first transformation is always horizontal and second transformation is always vertical

            for (int i = 0; i < 2; ++i) {
                for (int j = 0; j < 2; ++j) {
                    if (abs(k1[i][j] - k2[i][j]) > epsilon) return false;
                }
            }
            return true;
        });

    // sort by lattices per group
    std::sort(lattice_groups.begin(), lattice_groups.end(),
        [](const std::vector<Lattice>& g1, const std::vector<Lattice>& g2) {
            return g1.size() > g2.size(); });

#ifdef DEBUG_SAVE_LATTICE_GROUPS
    //visualize the lattice groups
    int group = 0;
    for (const std::vector<Lattice>& lattices : lattice_groups) {
        Mat clusterImg = image.clone();
        for (const Lattice& lattice : lattices) {
            for (const auto& grid_element : lattice.gridElements) {
                for (const auto& element_row : grid_element) {
                    for (const int element : element_row) {
                        const MSEREllipse& f = ellipses[element];
                        RotatedRect rect = f.brect;
                        Vec3b color(0, 0, 255); // BGR
                        cv::ellipse(clusterImg, f.brect, color);
                    }
                }
            }
        }
        char buffer[1024];
        sprintf(buffer, "%s%s%s_clusterGroup%d.png", baseOutputDir.c_str(), PATH_SEPARATOR, imageName.c_str(), group++);
        imwrite(buffer, clusterImg);
    }
#endif

    std::vector<ElementCluster> finalGrids;
    int finalGridInd = -1;
    int maxTotalFeatures = -1;
    float maxNormalizedCrossCorrScore;

    //for each bin of grids that have a similar transformation,
    //we will group their supporting features into the elements
    //of a common global grid

#ifdef VERBOSE
    printf("%d lattice groups detected!\n", lattice_groups.size());
#endif

    cv::Mat floatImage;
    image.convertTo(floatImage, CV_32FC3);
    floatImage /= 255.0f;
    for (int latticeInd = 0; latticeInd < lattice_groups.size(); latticeInd++) {
        //printf("latticeInd:%d\n", latticeInd);
        const std::vector<Lattice>& lattices = lattice_groups[latticeInd];

#ifdef MY_DEBUG
        printf("******Group\n*****");
        for (int l = 0; l < lattices.size(); l++) {
            printf("Lattice %d\n", lattices[l].latticeIndex);
        }
#endif

        //we will cluster the inlier features of each lattice in this group
        //such that each cluster of features represent one element of the
        //global grid the group suggests
        //If the global grid of the group has too many missing elements,
        //we will discard the group
        ElementCluster ec;
        if (!GraphCutGridElements(ellipses, lattices, image, ec))
            continue;

        //find the offset, i.e., the actual location where the elements of the grid will be separated
        if (findLatticeOffset(image, ec)) {
            if (norm(ec.firstGridTrans) < imgDistanceThreshold ||
                norm(ec.secondGridTrans) < imgDistanceThreshold)
                continue;

            //boundaries of the grid in the image
            cv::Vec2f gridMinPt((float)image.cols, (float)image.rows);
            cv::Vec2f gridMaxPt(0.0f, 0.0f);

            //the pixel location of each element of the grid will be stored in elementPosInImg
            ec.elementPosInImg.resize(ec.nCols);
            ec.elementPosInImg_nonClipped.resize(ec.nCols);
            //the pairwise similarity score for each grid element will be stored in matchingScores
            ec.matchingScores.resize(ec.nCols);

            //find the image region corresponding to each grid element
            for (int c = 0; c < ec.nCols; c++) {
                ec.elementPosInImg[c].resize(ec.nRows, Rect2f(0, 0, 0, 0));
                ec.elementPosInImg_nonClipped[c].resize(ec.nRows, Rect2f(0, 0, 0, 0));

                ec.matchingScores[c].resize(ec.nRows, 0.0f);
                for (int r = 0; r < ec.nRows; r++) {
                    cv::Vec2f startPt = ec.offset + ec.firstGridTrans*c + ec.secondGridTrans*r;
                    cv::Vec2f endPt = startPt + ec.firstGridTrans + ec.secondGridTrans;

                    ec.elementPosInImg_nonClipped[c][r] = Rect2f(startPt(0), startPt(1), endPt(0) - startPt(0), endPt(1) - startPt(1));

                    //if start or end point falls outside the image region, we will do clipping
                    if (startPt(0) < 0)
                        startPt(0) = 0.0f;
                    if (endPt(0) > image.cols - 1.0f)
                        endPt(0) = image.cols - 1.0f;
                    if (startPt(1) < 0)
                        startPt(1) = 0.0f;
                    if (endPt(1) > image.rows - 1.0f)
                        endPt(1) = image.rows - 1.0f;

                    float width = endPt(0) - startPt(0);
                    float height = endPt(1) - startPt(1);

                    if (width < 0 || height < 0) {
                        //printf("why is it happening\n");
                        width = 0;
                        height = 0;
                    }

                    ec.elementPosInImg[c][r] = Rect2f(startPt(0), startPt(1), width, height);

                    if (startPt(0) < gridMinPt(0))
                        gridMinPt(0) = startPt(0);
                    if (startPt(0) > gridMaxPt(0))
                        gridMaxPt(0) = startPt(0);

                    if (endPt(0) < gridMinPt(0))
                        gridMinPt(0) = endPt(0);
                    if (endPt(0) > gridMaxPt(0))
                        gridMaxPt(0) = endPt(0);

                    if (startPt(1) < gridMinPt(1))
                        gridMinPt(1) = startPt(1);
                    if (startPt(1) > gridMaxPt(1))
                        gridMaxPt(1) = startPt(1);

                    if (endPt(1) < gridMinPt(1))
                        gridMinPt(1) = endPt(1);
                    if (endPt(1) > gridMaxPt(1))
                        gridMaxPt(1) = endPt(1);
                }
            }

            //if area of the grid is less than half of the image, we discard it
            float areaRatio = (gridMaxPt(0) - gridMinPt(0)) / image.cols * (gridMaxPt(1) - gridMinPt(1)) / image.rows;

#ifdef MY_DEBUG
            printf("Area ratio of the grid:%f\n", areaRatio);
#endif

            if (areaRatio < MIN_GRID_AREA_RATIO)
                continue;

            //check if the base elements are actually similar by comparing the image regions to compute an overall matching score
            //overall matching score of the grid
            float matchingScore = 0;
            int noScores = 0;

            //per element matching score
            std::vector<std::vector<int> > scoresPerElement;
            scoresPerElement.resize(ec.nCols);
            for (int c = 0; c < ec.nCols; c++)
                scoresPerElement[c].resize(ec.nRows, 0);

            for (int b = 0; b < ec.nCols*ec.nRows; b++) {
                int row1 = b / ec.nCols;
                int col1 = b % ec.nCols;
                Rect r1 = ec.elementPosInImg[col1][row1];
                if (r1.width == 0 && r1.height == 0)
                    continue;

                for (int b2 = b + 1; b2 < ec.nCols*ec.nRows; b2++) {
                    int row2 = b2 / ec.nCols;
                    int col2 = b2 % ec.nCols;

                    Rect r2 = ec.elementPosInImg[col2][row2];
                    if (r2.width == 0 && r2.height == 0)
                        continue;

                    float score;
                    cv::Mat res;

                    if (r1.width != r2.width || r1.height != r2.height) {
                        //at least one of the elements falls outside the image
                        //find the common area between the two
                        Rect r1_nonClipped = ec.elementPosInImg_nonClipped[col1][row1];
                        Rect r2_nonClipped = ec.elementPosInImg_nonClipped[col2][row2];

                        float minX = r1_nonClipped.x;
                        if (r2_nonClipped.x < minX)
                            minX = r2_nonClipped.x;

                        float minY = r1_nonClipped.y;
                        if (r2_nonClipped.y < minY)
                            minY = r2_nonClipped.y;

                        float maxX = r1_nonClipped.x + r1_nonClipped.width;
                        if (r2_nonClipped.x + r2_nonClipped.width > maxX)
                            maxX = r2_nonClipped.x + r2_nonClipped.width;

                        float maxY = r1_nonClipped.y + r1_nonClipped.height;
                        if (r2_nonClipped.y + r2_nonClipped.height > maxY)
                            maxY = r2_nonClipped.y + r2_nonClipped.height;

                        if (minX < 0) {
                            r1_nonClipped.x -= minX;
                            r2_nonClipped.x -= minX;
                            r1_nonClipped.width += minX;
                            r2_nonClipped.width += minX;
                        }
                        if (minY < 0) {
                            r1_nonClipped.y -= minY;
                            r2_nonClipped.y -= minY;
                            r1_nonClipped.height += minY;
                            r2_nonClipped.height += minY;
                        }

                        if (maxX >= image.cols) {
                            r1_nonClipped.width -= (maxX - image.cols);
                            r2_nonClipped.width -= (maxX - image.cols);

                        }
                        if (maxY >= image.rows) {
                            r1_nonClipped.height -= (maxY - image.rows);
                            r2_nonClipped.height -= (maxY - image.rows);
                        }

                        //printf("r1:%d %d %d %d %d %d, r2:%d %d %d %d %d %d\n", r1_nonClipped.x, r1_nonClipped.y, r1_nonClipped.x+r1_nonClipped.width, r1_nonClipped.y+r1_nonClipped.height, r1_nonClipped.width, r1_nonClipped.height, r2_nonClipped.x, r2_nonClipped.y, r2_nonClipped.x+r2_nonClipped.width, r2_nonClipped.y+r2_nonClipped.height, r2_nonClipped.width, r2_nonClipped.height);
                        matchTemplate(floatImage(r1_nonClipped), floatImage(r2_nonClipped), res, CV_TM_CCORR_NORMED);
                        score = res.at<float>(0, 0);
                    } else {
                        //float score = cv::norm(floatImage(r1)-floatImage(r2));
                        //score /= float(r1.width*r1.height);
                        //score = 1.0 - score;

                        matchTemplate(floatImage(r1), floatImage(r2), res, CV_TM_CCORR_NORMED);
                        score = res.at<float>(0, 0);
                    }

                    ec.matchingScores[col1][row1] = score;
                    scoresPerElement[col1][row1] += 1;
                    ec.matchingScores[col2][row2] += score;
                    scoresPerElement[col2][row2] += 1;

                    matchingScore += score;
                    noScores += 1;
                }
            }

            for (int c = 0; c < ec.nCols; c++) {
                for (int r = 0; r < ec.nRows; r++) {
                    if (scoresPerElement[c][r] > 0)
                        ec.matchingScores[c][r] = ec.matchingScores[c][r] / (float)scoresPerElement[c][r];
                }
            }

#ifdef MY_DEBUG
            printf("Average norm cross corr score:%f\n", matchingScore / noScores);
#endif MY_DEBUG

            //if the elements are not similar based on image similarity, we will discard the current grid
            if (noScores > 0 && matchingScore / noScores < MIN_MATCHING_SCORE) {
                //printf("Lattice discarded!\n");
                continue;
            }

            //count the number of MSER features that contributed to this final grid
            int totalFeatures = 0;
            for (int j = 0; j < lattices.size(); j++) {
                for (int c = 0; c < lattices[j].gridElements.size(); c++) {
                    for (int r = 0; r < lattices[j].gridElements[c].size(); r++) {
                        if (lattices[j].gridElements[c][r].size() > 0)
                            totalFeatures += 1;
                    }
                }
            }

            //we will look at the total number of features supporting the grid
            if (totalFeatures < MIN_TOTAL_FEATURES_GRID)
                continue;

            finalGrids.push_back(ec);
            if (finalGridInd == -1) {
                finalGridInd = finalGrids.size() - 1;
                maxTotalFeatures = totalFeatures;
            } else if (totalFeatures > maxTotalFeatures) {
                finalGridInd = finalGrids.size() - 1;
                maxTotalFeatures = totalFeatures;
            }
        }
    }

    if (finalGridInd != -1) {
        ElementCluster ec = finalGrids[finalGridInd];
		#ifdef DEBUG_BULK_PATTERN_EXTRACTION_RESULTS
			Mat newImg = image.clone();
			char buffer[1024];
			sprintf(buffer, "%s%s%s_baseElement.png", baseOutputDir.c_str(), PATH_SEPARATOR, imageName.c_str());
		#endif

        //save out the base element
        //the base element is the one that has the highest matching score with the other elements
        //if such an element is not found, we will choose the base element which lies inside the image boundary
        bool baseElementFound = false;
        float maxMatchingScore = 0;
        Vec2i baseElementInd(0, 0);
        for (int c = 0; c < ec.nCols; c++) {
            for (int r = 0; r < ec.nRows; r++) {
                if (ec.matchingScores[c][r] > maxMatchingScore &&
                    fabs(ec.elementPosInImg[c][r].width - ec.firstGridTrans[0]) < 1.0 &&
                    fabs(ec.elementPosInImg[c][r].height - ec.secondGridTrans[1]) < 1.0) {
                    maxMatchingScore = ec.matchingScores[c][r];
                    baseElementInd(0) = c;
                    baseElementInd(1) = r;
                }
            }
        }
        if (maxMatchingScore > MIN_MATCHING_SCORE) {
            baseElementFound = true;
        }

        if (!baseElementFound) {
            for (int c = 0; c < ec.nCols; c++) {
                for (int r = 0; r < ec.nRows; r++) {
                    if (fabs(ec.elementPosInImg[c][r].width - ec.firstGridTrans[0]) < 1.0 &&
                        fabs(ec.elementPosInImg[c][r].height - ec.secondGridTrans[1]) < 1.0) {
                        baseElementInd(0) = c;
                        baseElementInd(1) = r;
                        baseElementFound = true;
                        break;
                    }
                }
                if (baseElementFound)
                    break;
            }
        }

        //save out the final grid
        if (baseElementFound) {
            cv::Mat baseElement(image(ec.elementPosInImg[baseElementInd(0)][baseElementInd(1)]));
			basePattern = baseElement;

			#ifdef DEBUG_BULK_PATTERN_EXTRACTION_RESULTS
				cv::imwrite(buffer, baseElement);
				/*if (baseElement.rows < 50 || baseElement.cols < 50) {
				float scale_parameter = 50.0 / min(baseElement.rows, baseElement.cols);
				float width = baseElement.cols * scale_parameter;
				float height = baseElement.rows * scale_parameter;
				cv::resize(baseElement, baseElement, cv::Size(width, height), 0, 0);
				}
				cv::imwrite(buffer, baseElement);
				*/
				for (int c = 0; c <= ec.nCols; c++) {
					cv::Vec2f startPt = ec.offset + ec.firstGridTrans*c;
					cv::Vec2f endPt = startPt + ec.secondGridTrans*ec.nRows;
					cv::line(newImg, cv::Point2f(startPt(0), startPt(1)),
						cv::Point2f(endPt(0), endPt(1)),
						Vec3b(255, 255, 0), 2.0f);
				}
				for (int r = 0; r <= ec.nRows; r++) {
					cv::Vec2f startPt = ec.offset + ec.secondGridTrans*r;
					cv::Vec2f endPt = startPt + ec.firstGridTrans*ec.nCols;
					cv::line(newImg, cv::Point2f(startPt(0), startPt(1)),
						cv::Point2f(endPt(0), endPt(1)),
						Vec3b(255, 255, 0), 2.0f);
				}
				sprintf(buffer, "%s%s%s_finalGrid.png", baseOutputDir.c_str(), PATH_SEPARATOR, imageName.c_str());
				//cout << buffer;
				cv::imwrite(buffer, newImg);
			#endif
        }
    }
	else {
		#ifdef DEBUG_BULK_PATTERN_EXTRACTION_RESULTS
			noOut.push_back(imageName); // No Pattern Found case: 
		#endif
	}

#if 0
    namedWindow("test window", WINDOW_GUI_NORMAL);
    Point2f offset(0.0f - tmp.rows, 0.0f - tmp.cols);
    Mat test = Mat::zeros(tmp.rows * 2, tmp.cols * 2, CV_8UC1);
    for (int i = 0; i < ellipses.size(); ++i) {
        size_t label = labels[i];
        if (clusters[label].size() <= 3) continue;
        const MSEREllipse& e = ellipses[i];
        RotatedRect rect = e.brect;
        rect.center -= e.center + offset;
        cv::ellipse(test, rect, 255);
        Vec3b color = hashInt(labels[i]);
        color = Vec3b(255 - color(0), 255 - color(1), 255 - color(2));
        cv::ellipse(test, rect, color);

        cv::ellipse(image, e.brect, color);
    }

    //applyColorMap(test, test, cv::COLORMAP_JET);
    imshow("test window", test);

    namedWindow("Display window", WINDOW_GUI_NORMAL);// Create a window for display.
    imshow("Display window", image);                   // Show our image inside it.

    waitKey(0);
#endif

    return finalGridInd != -1;
}
