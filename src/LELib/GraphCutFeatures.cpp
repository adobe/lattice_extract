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

#include <opencv2/highgui/highgui.hpp>

#pragma warning(push)
#pragma warning(disable:4267) // size_t to int conversion
#include <andres/graph/graph.hxx>
#include <andres/graph/multicut-lifted/greedy-additive.hxx>
#include <andres/graph/multicut-lifted/kernighan-lin.hxx>
#pragma warning(pop)

#include "Utils.h"
#include "CommonParameters.h"

#ifdef USE_TBB
#include <tbb/tbb.h>
#endif

//#define DEBUG_SAVE_SIMILARITY_MATRICES

using namespace andres::graph::multicut_lifted;
using namespace std;

namespace {
    float GetWeight(float shapeDist, float appDist) {
        float shape, appearence;
        {
            constexpr float app_min = 3.0; // below this it's a perfect match
            constexpr float app_bad = 5.0; // above this is's a bad match
            constexpr float app_range = app_bad - app_min;

            // first we move x left, such that it's at 0 if good.
            // this is essentially a function of x, where x = appDist. 
            // We know that x can be 0 or greater.
            float x = std::max(appDist - app_min, 0.0f);

            // this is a gaussian centered at 0, without the scaling component, 
            // so we blend from f(0)=1 to almost 0 at x=app_range
            float g = Gaussian(x, app_range / 2);

            // this is the component that will drive bad matches to weights with negative values.
            // it is an exponential centered at (app_range), but we only take the right part. 
            // exponent and scaling control the slope.
            constexpr float exponent = 3.0f;
            constexpr float scaling = 5.0f;
            float c = -pow(std::max(x - app_range, 0.0f), exponent) / scaling;

            // and then we just sum them. gaussian dominates before (app_range), 
            // exponential dominates after.
            appearence = g + c;
        }

        {
            // shape is similar, but we know it has to be in (0, 1) range.
            // 0->1 directly maps to the difference in size, e.g. a weight of .2
            // means that the smaller one is 20% smaller of the bigger one 
            // (or 80% of its size) in one axis.
            //
            // we just add two gaussians, a "good" one centered at 0, 
            // and a "bad" one centered at 1, scaled by -20.
            float x = shapeDist;
            shape = Gaussian(x, .01) - 20.0f * Gaussian(x, 1, .05f);
        }

        // taking the worse weight of the two, but no lower than -50, to avoid making
        // the graphcut algorithm go crazy over huge gains.
        return std::max(std::min(appearence, shape), -50.0f);
    }
}

//graph cut for ellipses which are defined by ellipseIds
void GraphCutFeatures(std::vector<MSEREllipse>& ellipses,
                      const std::vector<unsigned long> ellipseIds,
                      std::vector<std::vector<unsigned long>> &final_clusters,
                      const cv::Mat& image){
    
#ifdef VERBOSE
    printf("Graph cut: building graph\n");
#endif
    
    int imgCols = image.cols;
    int imgRows = image.rows;
    
    // each ellipse is a vertex in the graph
    andres::graph::Graph<> graph;
    unsigned long n_ellipses = (unsigned long)ellipseIds.size();
    graph.insertVertices(n_ellipses);
    
    vector<float> weights; // size = number of edges in graph
    
    vector<std::pair<unsigned long, unsigned long> > edgeIndices;
    
    for (unsigned long i = 0; i < n_ellipses; ++i) {
        
        const MSEREllipse& e1 = ellipses[ellipseIds[i]];
        
        for (unsigned long j = i + 1; j < n_ellipses; ++j) {
            const MSEREllipse& e2 = ellipses[ellipseIds[j]];
            
            if (MSEREllipse::CanMakeALink(e1, e2, imgCols, imgRows)) {
                
                graph.insertSingleEdge(i, j);
                /*float shape = MSEREllipse::shapeDissimilarity(ellipses[i], ellipses[j]);
                 float shapeDist = Gaussian(shape, .01) - 20.0f * Gaussian(shape, 1, .05f);
                 float weight = 0.0f;
                 if(shapeDist < 0.0f)
                 weight = shapeDist;
                 else {
                 float appearence = MSEREllipse::appearanceSimilarity(ellipses[i], ellipses[j]);
                 weight = (float)GetWeight(shape, appearence);
                 }
                 weights.push_back(weight);*/
            }
        }
    }
    
    //compute edge weights
    unsigned long nEdges = graph.numberOfEdges();
    weights.resize(nEdges);
#ifdef USE_TBB
    tbb::parallel_for((unsigned long)(0), (unsigned long)(nEdges), [&](unsigned long index) {
#else
    for (int index = 0; index < nEdges; index++) {
#endif
        unsigned long v1, v2;
        graph.getVerticesOfEdge(index, v1, v2);
            
        float shape = MSEREllipse::shapeDissimilarity(ellipses[ellipseIds[v1]], ellipses[ellipseIds[v2]]);
        float shapeDist = Gaussian(shape, .01) - 20.0f * Gaussian(shape, 1, .05f);
        float weight = 0.0f;
        if (shapeDist < 0.0f)
                weight = shapeDist;
        else {
            float appearence = MSEREllipse::appearanceSimilarity(ellipses[ellipseIds[v1]], ellipses[ellipseIds[v2]]);
            weight = (float)GetWeight(shape, appearence);
        }
        weights[index] = (weight);
#ifdef USE_TBB
    });
#else
    }
#endif
                      
                      
    assert(graph.numberOfEdges() == weights.size());
                      
#ifdef VERBOSE
    printf("Graph has %lu nodes and %lu edges\n", graph.numberOfVertices(), graph.numberOfEdges());
                      
    printf("Graph cut: computing... ");
#endif
                      
    clock_t start = clock();
    vector<unsigned long> tmp_labels = greedyAdditiveEdgeContraction(graph, graph, weights);
                      
#ifdef VERBOSE
    printf("done (%f s)\n", (clock() - start) / (double)CLOCKS_PER_SEC);
    printf("Graph cut: computing...");
#endif
                      
    vector<unsigned long> vertex_labels = kernighanLin(graph, graph, weights, tmp_labels);
                      
#ifdef VERBOSE
    printf("done (%f s)\n", (clock() - start) / (double)CLOCKS_PER_SEC);
#endif
                      
    // construct clusters vector from labels
    assert(vertex_labels.size() == ellipseIds.size());
    size_t maxlabel = *std::max_element(vertex_labels.begin(), vertex_labels.end());
                      
    // lists of indexes to ellipses/labels vector
    int startIndex = final_clusters.size();
    for(int c=0; c<maxlabel+1; c++)
        final_clusters.push_back(std::vector<unsigned long>());
    for (int i = 0; i < vertex_labels.size(); ++i){
    final_clusters[startIndex+vertex_labels[i]].push_back(ellipseIds[i]);
    }
                      
    // remove small clusters
    final_clusters.erase(remove_if(final_clusters.begin()+startIndex,
                             final_clusters.end(),
                             [](const vector<unsigned long>& c) {
                                return c.size() <= DISCARD_CLUSTERS_SMALLER_THAN;
                                               }), final_clusters.end());
                      
}

std::vector<std::vector<unsigned long>> GraphCutFeatures(
	std::vector<MSEREllipse>& ellipses,
	const cv::Mat& image) {

    int imgCols = image.cols;
    int imgRows = image.rows;
    
#ifdef DEBUG_SAVE_SIMILARITY_MATRICES
	cv::Size matSize(nEllipses, nEllipses);
	cv::Mat1f shapeDists = cv::Mat1f::zeros(matSize);
	cv::Mat1f appDists = cv::Mat1f::zeros(matSize);

	cout << "Computing MSER similarity matrices" << endl;
	float maxDist = 0;
	for (int i = 0; i < ellipses.size(); ++i) {
		for (int j = i + 1; j < ellipses.size(); ++j) {

            if (MSEREllipse::CanMakeALink(ellipses[i], ellipses[j], imgCols, imgRows)) {
				// note: we are building a triangular matrix, but that's ok, since we're doing
				// lookup in the same way. Also, only compute the ones that we actually need.
				float shapeDist = 0.0f;
				shapeDists.at<float>(i, j) = shapeDist;

				float appearDist = MSEREllipse::appearanceSimilarity(ellipses[i], ellipses[j]);
				appDists.at<float>(i, j) = appearDist;
			}
		}
	}

	cv::Mat display;
	float maxShapeDist;
	cv::minMaxLoc(shapeDists, nullptr, &maxShapeDist);
	shapeDists.convertTo(display, CV_8UC1, 255.0 / maxShapeDist, 0);
	applyColorMap(display, display, cv::COLORMAP_JET);
	cv::imwrite("distances_shape.png", display);

	float maxAppearDist;
	cv::minMaxLoc(appDists, nullptr, &maxAppearDist);
	appDists.convertTo(display, CV_8UC1, 255.0 / maxAppearDist, 0);
	applyColorMap(display, display, cv::COLORMAP_JET);
	cv::imwrite("distances_appearence.png", display);
#endif

#ifdef VERBOSE
	printf("Graph cut: building graph\n");
#endif

	// each ellipse is a vertex in the graph
	andres::graph::Graph<> graph;
	unsigned long n_ellipses = (unsigned long)ellipses.size();
	graph.insertVertices(n_ellipses);

	vector<float> weights; // size = number of edges in graph

	vector<std::pair<unsigned long, unsigned long> > edgeIndices;

	clock_t begin = clock();

	for (unsigned long i = 0; i < n_ellipses; ++i) {

		const MSEREllipse& e1 = ellipses[i];

		for (unsigned long j = i + 1; j < n_ellipses; ++j) {
			const MSEREllipse& e2 = ellipses[j];

            if (MSEREllipse::CanMakeALink(e1, e2, imgCols, imgRows)) {

				graph.insertSingleEdge(i, j);
				/*float shape = MSEREllipse::shapeDissimilarity(ellipses[i], ellipses[j]);
				float shapeDist = Gaussian(shape, .01) - 20.0f * Gaussian(shape, 1, .05f);
				float weight = 0.0f;
				if(shapeDist < 0.0f)
					weight = shapeDist;
				else {
					float appearence = MSEREllipse::appearanceSimilarity(ellipses[i], ellipses[j]);
					weight = (float)GetWeight(shape, appearence);
				}
				weights.push_back(weight);*/
			}
		}
	}

    //compute edge weights
	unsigned long nEdges = graph.numberOfEdges();
	weights.resize(nEdges);
#ifdef USE_TBB
	tbb::parallel_for((unsigned long)(0), (unsigned long)(nEdges), [&](unsigned long index) {
#else
	for (int index = 0; index < nEdges; index++) {
#endif
		unsigned long v1, v2;
		graph.getVerticesOfEdge(index, v1, v2);

		float shape = MSEREllipse::shapeDissimilarity(ellipses[v1], ellipses[v2]);
		float shapeDist = Gaussian(shape, .01) - 20.0f * Gaussian(shape, 1, .05f);
		float weight = 0.0f;
		if (shapeDist < 0.0f)
			weight = shapeDist;
		else {
			float appearence = MSEREllipse::appearanceSimilarity(ellipses[v1], ellipses[v2]);
			weight = (float)GetWeight(shape, appearence);
		}
		weights[index] = (weight);
#ifdef USE_TBB
	});
#else
    }
#endif
    
    assert(graph.numberOfEdges() == weights.size());
                      
#ifdef VERBOSE
    printf("Graph has %lu nodes and %lu edges\n", graph.numberOfVertices(), graph.numberOfEdges());
    
    printf("Graph cut: computing... ");
#endif
                      
    clock_t start = clock();
    vector<unsigned long> tmp_labels = greedyAdditiveEdgeContraction(graph, graph, weights);
                      
#ifdef VERBOSE
    printf("done (%f s)\n", (clock() - start) / (double)CLOCKS_PER_SEC);
    printf("Graph cut: computing...");
#endif
                      
    vector<unsigned long> vertex_labels = kernighanLin(graph, graph, weights, tmp_labels);
                      
#ifdef VERBOSE
    printf("done (%f s)\n", (clock() - start) / (double)CLOCKS_PER_SEC);
#endif
                      
    // construct clusters vector from labels
    assert(vertex_labels.size() == ellipses.size());
    size_t maxlabel = *std::max_element(vertex_labels.begin(), vertex_labels.end());

    // lists of indexes to ellipses/labels vector
    vector<vector<unsigned long>> clusters(maxlabel + 1, vector<unsigned long>());
    for (int i = 0; i < vertex_labels.size(); ++i) {
        clusters[vertex_labels[i]].push_back(i);
    }

    // remove single-element clusters
    clusters.erase(remove_if(clusters.begin(), clusters.end(),
        [](const vector<unsigned long>& c) {
            return c.size() <= DISCARD_CLUSTERS_SMALLER_THAN;
        }), clusters.end());

    // sort clusters by size
    sort(clusters.begin(), clusters.end(), 
        [](const vector<unsigned long>& c1, const vector<unsigned long>& c2) {
            return c1.size() > c2.size();
        });

    return clusters;
}
