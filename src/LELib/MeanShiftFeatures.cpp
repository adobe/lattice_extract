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

#include "GraphCutFeatures.h"

#include <ctime>
#include <opencv2/core/core.hpp>

#include <ctime>
#include <iostream>

#include "EllipseVPTree.h"
#include "CommonParameters.h"

#ifdef USE_TBB
#include <tbb/tbb.h>
#endif

//#define MS_VALIDATE_LINEAR

#define MAX_ITERATOR_COUNT 100000

using namespace std;

namespace {
    
    float computeThreshold(float shapeSimilarityWeight, float apperanceSimilarityWeight){
        float thresh = 4.0f;
        if(shapeSimilarityWeight == 0.0f)
            thresh = 4.0f;
        else if(apperanceSimilarityWeight == 0.0f)
            thresh = 1.5f;
        //todo:else
        
        return thresh;
    }
    
	//structure to represent assignments of features to clusters for bookkeeping purposes
	struct ClusterAssignment {
		size_t featureIndex;
		size_t clusterIndex;

		ClusterAssignment(size_t index):featureIndex(index),clusterIndex(index){}

		bool operator < (const ClusterAssignment & other) const {
			return clusterIndex < other.clusterIndex;

		}
	};

	class MeanEllipse {
	public:
		cv::Size2f size;
		cv::Mat img;
		size_t parentIndex;
		bool converged;
		bool merged;
        
        static float shapeWeight;
        static float appearanceWeight;

        MeanEllipse(){}

		MeanEllipse(const MSEREllipse & src, size_t parent):size(src.brect.size),img(src.normImg),parentIndex(parent),converged(false),merged(false){}

		float shapeDissimilarity(const MSEREllipse & e) {
			cv::Size2f se = e.brect.size;
			return MSEREllipse::shapeDissimilarity(size, se);

		}


		float appearanceDissimilarity(const MSEREllipse & e ) {
			return MSEREllipse::appearanceSimilarity(img, e.normImg);
		}

		float dist(const MSEREllipse & e) {
            if(shapeWeight == 0.0f)
                return appearanceDissimilarity(e);
            else if(appearanceWeight == 0.0f)
                return shapeDissimilarity(e);
            else
                return shapeWeight*shapeDissimilarity(e) + appearanceWeight*appearanceDissimilarity(e);
		}
		float dist(const MeanEllipse & e) {
            if(shapeWeight == 0.0f)
                return MSEREllipse::appearanceSimilarity(img, e.img);
            else if(appearanceWeight == 0.0f)
                return MSEREllipse::shapeDissimilarity(size, e.size);
            else
                return shapeWeight*MSEREllipse::shapeDissimilarity(size, e.size) + appearanceWeight*MSEREllipse::appearanceSimilarity(img, e.img);
		}

		float update(cv::Size2f& newSz, cv::Mat & newImg) {
			float shapedist = MSEREllipse::shapeDissimilarity(size, newSz);
			float appdist = MSEREllipse::appearanceSimilarity(img, newImg);

			size = newSz;
			img = newImg;

            return shapeWeight*shapedist + appearanceWeight*appdist;
		}
	};
    
    float MeanEllipse::shapeWeight = 1.0f;
    float MeanEllipse::appearanceWeight = 1.0f;

	float epanechnikov(float u, float width = 1.f) {
		return (0.75*(1-pow(u/width,2))) / width;
	}
}



std::vector<std::vector<unsigned long>> MeanShiftFeatures(
	int& outMaxIterCount,
	const std::vector<MSEREllipse>& ellipses,
	const cv::Mat& image, float shapeWeight, float appearanceWeight, bool treeSearch) {
		
	outMaxIterCount = -1;

	//timing
#ifdef VERBOSE
	printf("Mean shift: computing... ");
#endif
	clock_t start = clock();
	
    MeanEllipse::shapeWeight = shapeWeight;
    MeanEllipse::appearanceWeight = appearanceWeight;
    MSEREllipse::shapeWeight = shapeWeight;
    MSEREllipse::apperanceWeight = appearanceWeight;
    
	//initialise the cluster assignments vector
	vector<ClusterAssignment> assignments;
	assignments.reserve(ellipses.size());
	for (size_t i = 0; i < ellipses.size(); ++i) {
		assignments.push_back(ClusterAssignment(i));
	}
    
	//initialise the means vector
	vector<MeanEllipse> means;
	means.reserve(ellipses.size());
	for (size_t i = 0; i < ellipses.size(); ++i) {
		means.push_back(MeanEllipse(ellipses[i],i));
	}
   
	EllipseVPTree<MSEREllipse> eTree(ellipses);
    
	//mean is converged if it moves less than this in an iteration
	const float convergedThreshold = 1e-3f;
	//mean is merged with another if they are closer than this threshold
	const float mergeThreshold = 1e-2f;
	//threshold for the distance search/kernel
    float thresh = computeThreshold(shapeWeight, appearanceWeight);

	bool converged = false;
	size_t numConverged = 0;
	int iterationCtr = 1;
    
    int convergedCount = 0;

	do{
        
        //run a mean shift iteration
#ifdef USE_TBB
		tbb::parallel_for((size_t)(0), (size_t)(means.size()-numConverged), [&](size_t index) {
#else
        for (size_t index = 0; index < means.size()-numConverged; index++) {
#endif
            
			float sumWeight = 0.f;
			cv::Mat nextImg = cv::Mat3f::zeros(means[index].img.size());
			cv::Size2f nextSize(0.f,0.f);
			MeanEllipse & mean = means[index];
			int updateCtr = 0;
            
            if(!treeSearch) {
                eTree.rangeSearchLinear(mean, [&](const MSEREllipse & e, float dist, size_t index) {
                    float w= epanechnikov(dist, thresh);
                    nextImg += e.normImg*w;
                    nextSize += e.brect.size*w;
                    sumWeight += w;
                    ++updateCtr;
                    }, thresh);
            }
            else
            {
                eTree.rangeSearch(mean, [&](const MSEREllipse & e, float dist, size_t index) {
                    float w = epanechnikov(dist, thresh);
                    nextImg += e.normImg*w;
                    nextSize += e.brect.size*w;
                    sumWeight += w;
                    ++updateCtr;
                }, thresh);
            }

			if (1e-6 >= sumWeight) {
				mean.converged = true;
				convergedCount += 1;
			} else {
				nextImg /= sumWeight;
				nextSize /= sumWeight;

	 			float change = mean.update(nextSize, nextImg);
            
				if (change < convergedThreshold) {
					mean.converged = true;
					convergedCount += 1;
				}
			}


#ifdef USE_TBB
		});
#else
        }
#endif
        EllipseVPTree<MeanEllipse> mergeTree(means);
        //run a proximity check to merge means that are too close
		//currently brute force, might be replaced with VP-tree later on
		for (auto it = means.begin(); it != means.end()-numConverged; ++it) {
			MeanEllipse & refMean = *it;
            
            if(!treeSearch)
            {
                mergeTree.rangeSearchLinearTerminated(refMean, [&](const MeanEllipse& mergeMean, float dist, size_t index) {
                    
                    if (mergeMean.parentIndex == refMean.parentIndex) {
                        //do not merge with self
                        return false;
                    }
                    if (mergeMean.merged) {
                        //do not merge with already merged
                        return false;
                    }
                    //assign all the features belonging to refMean's cluster into the mergeMean's cluster
                    auto range = std::equal_range(assignments.begin(), assignments.end(), ClusterAssignment(refMean.parentIndex));
                    auto mergeRange = std::equal_range(assignments.begin(), assignments.end(), ClusterAssignment(mergeMean.parentIndex));
                    for (auto assignIt = range.first; assignIt != range.second; ++assignIt) {
                        assignIt->clusterIndex = mergeMean.parentIndex;
                    }
                    //re-sort the range between refMean's and mergeMean's cluster
                    //sanity check: mergeRange should always be after range
                    sort(min(range.first, mergeRange.first), max(range.second, mergeRange.second));
                    //tag refMean
                    refMean.merged = true;
                    
                    return true;
                }, mergeThreshold);
            }
            else {
                mergeTree.rangeSearchTerminated(refMean, [&](const MeanEllipse & mergeMean, float dist, size_t index) {
                
                    if (mergeMean.parentIndex == refMean.parentIndex) {
                        //do not merge with self
                        return false;
                    }
                    if (mergeMean.merged) {
                        //do not merge with already merged
                        return false;
                    }
                    //assign all the features belonging to refMean's cluster into the mergeMean's cluster
                    auto range = std::equal_range(assignments.begin(), assignments.end(), ClusterAssignment(refMean.parentIndex));
                    auto mergeRange = std::equal_range(assignments.begin(), assignments.end(), ClusterAssignment(mergeMean.parentIndex));
                    for (auto assignIt = range.first; assignIt != range.second; ++assignIt) {
                        assignIt->clusterIndex = mergeMean.parentIndex;
                    }
                    //re-sort the range between refMean's and mergeMean's cluster
                    //sanity check: mergeRange should always be after range
                    sort(min(range.first, mergeRange.first), max(range.second, mergeRange.second));
                    //tag refMean
                    refMean.merged = true;
                
                    return true;
                }, mergeThreshold);
            }
        }

		//remove the merged means
		auto meansEnd = remove_if(means.begin(), means.end(), [](const MeanEllipse & e) {return e.merged; });
		if (meansEnd != means.end()) {
			means.erase(meansEnd, means.end());
		}		
		numConverged= means.end()-partition(means.begin(), means.end(), [](const MeanEllipse & m) {return !m.converged; });
		converged = (numConverged == means.size());
		
        ++iterationCtr;

		if (iterationCtr > outMaxIterCount)
		{
			outMaxIterCount = iterationCtr;
		}

		if (iterationCtr >= MAX_ITERATOR_COUNT)
		{
			return vector<vector<unsigned long>>();
		}

    } while (!converged);

	// cout << "iterationCtr: " << iterationCtr << "\n";

	vector<vector<unsigned long>> ret;
    
	vector<size_t> retainedMeans;

	//write out the clusters
	assert(is_sorted(assignments.begin(), assignments.end()));
	auto assignIt = assignments.begin();
	do {
		auto range = equal_range(assignIt, assignments.end(), ClusterAssignment(assignIt->clusterIndex));
		size_t clusterSize = range.second - range.first;
		if (clusterSize >= DISCARD_CLUSTERS_SMALLER_THAN)
        {
			ret.push_back(vector<unsigned long>());
			ret.back().reserve(clusterSize);
			for (auto it = range.first; it != range.second; ++it) {
				ret.back().push_back(it->featureIndex);
			}
			retainedMeans.push_back(assignIt->clusterIndex);
		}
		assignIt = range.second;		
	} while (assignIt != assignments.end());

	/*cout << "cluster affinity matrix: " << endl;
	for (int i = 0; i < retainedMeans.size(); ++i) {
		MeanEllipse & meanI = *find_if(means.begin(), means.end(), [&](const MeanEllipse & e) {return e.parentIndex == retainedMeans[i]; });
		for (int j = 0; j < retainedMeans.size(); ++j) {			
			MeanEllipse & meanJ = *find_if(means.begin(), means.end(), [&](const MeanEllipse & e) {return e.parentIndex == retainedMeans[j]; });
			cout << meanI.dist(meanJ) << "\t";
		}
		cout << endl;
	}*/

	sort(ret.begin(), ret.end(), [](const vector<unsigned long> & v1, const vector<unsigned long> & v2) {return v1.size() > v2.size(); });
	
#ifdef VERBOSE
	printf("done (%f s)\n", (clock() - start) / (double)CLOCKS_PER_SEC);
	printf("Mean Shift completed in %d iterations\n",iterationCtr);
	printf("Returning %d clusters\n", ret.size());
#endif

	return ret;

}

