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

template <typename RefEllipse, unsigned int maxPointsPerLeaf=2>
struct EllipseVPTree {
	struct Node {
		Node() {}
		Node(bool _isLeaf, int _lIndex, int _rIndex, float _radius = 0.f) :isLeaf(_isLeaf), lIndex(_lIndex), rIndex(_rIndex), radius(_radius) {}
		bool isLeaf;
		float radius;
		RefEllipse refEllipse;
		size_t refIndex;
		///during construction, these point to the data points vector. Thereafter these are child indices (-1 for none)
		int lIndex, rIndex;
	};

	EllipseVPTree(const std::vector<RefEllipse>& srcPoints):points(srcPoints) {
		for (size_t i = 0; i < srcPoints.size(); ++i) {
			pointIndices.push_back(std::make_pair(i, 0.f));
		}

		std::vector<size_t> nodeProcStack;
		nodes.push_back(Node(true, 0, points.size()));
		nodeProcStack.push_back(0);

		while (!nodeProcStack.empty()) {
			size_t procIndex = nodeProcStack.back();
			nodeProcStack.pop_back();

			assert(nodes[procIndex].isLeaf);
			Node & node = nodes[procIndex];

			int sz = node.rIndex - node.lIndex;

			if (sz > 2) {
				//further split
				//start by picking a pivot. for the time being, it will be the first point
				node.refEllipse = points[pointIndices[node.lIndex].first];
				node.refIndex = pointIndices[node.lIndex].first;
				//sort all the other ellipses by distance from here
				for (auto it = pointIndices.begin() + node.lIndex + 1; it != pointIndices.begin() + node.rIndex; ++it) {
					it->second = node.refEllipse.dist(points[it->first]);
				}
				std::sort(pointIndices.begin() + node.lIndex + 1, pointIndices.begin() + node.rIndex, [&](const std::pair<size_t,float> & p1, const std::pair<size_t, float> & p2) {return p1.second < p2.second; });
				//std::sort(pointIndices.begin() + node.lIndex + 1, pointIndices.begin() + node.rIndex, [&](const std::pair<size_t, float> & p1, const std::pair<size_t, float> & p2) {return node.refEllipse.dist(points[p1.first]) < node.refEllipse.dist(points[p2.first]); });
				//get the median distance
				size_t splitIndex = node.lIndex + 1 + (sz - 1) / 2;
				if ((sz-1)%2 == 0){					
					node.radius = (node.refEllipse.dist(points[pointIndices[splitIndex - 1].first]) + node.refEllipse.dist(points[pointIndices[splitIndex].first]))/2;
				}
				else {
					node.radius = node.refEllipse.dist(points[pointIndices[splitIndex].first]);
				}
				node.isLeaf = false;

				nodes.push_back(Node(true, nodes[procIndex].lIndex+1, splitIndex));
				nodes.push_back(Node(true, splitIndex, nodes[procIndex].rIndex));

				nodes[procIndex].lIndex = nodes.size() - 2;
				nodes[procIndex].rIndex = nodes.size() - 1;
				nodeProcStack.push_back(nodes[procIndex].lIndex);
				nodeProcStack.push_back(nodes[procIndex].rIndex);
			}
			else if (sz == 2) {
				//construct a single-child node
				node.refEllipse = points[pointIndices[node.lIndex].first];
				node.refIndex = pointIndices[node.lIndex].first;
				node.isLeaf = false;
				node.radius = node.refEllipse.dist(points[pointIndices[node.lIndex + 1].first]) + (1e-3f);
				

				nodes.push_back(Node(true, nodes[procIndex].lIndex + 1, nodes[procIndex].rIndex));
				
				nodes[procIndex].rIndex = -1;
				nodes[procIndex].lIndex = nodes.size()-1;
				nodeProcStack.push_back(nodes[procIndex].lIndex);
			}
			else if (sz == 1) {
				//construct a true leaf
				node.refEllipse = points[pointIndices[node.lIndex].first];
				node.refIndex = pointIndices[node.lIndex].first;
				node.isLeaf = true;
				node.radius = -1;
				node.rIndex = -1;
				node.lIndex = -1;
			} else if (sz<=0){
				//shouldn't be happening
				assert(false);
			}

			/*if ((node.rIndex - node.lIndex) <= maxPointsPerLeaf) {
				//no split
			}
			else {
				//for the time being, pick the first node as the reference point
				node.refEllipse = points[node.lIndex];
				std::sort(points.begin() + node.lIndex, points.begin() + node.rIndex, [&](const RefEllipse & e1, const RefEllipse & e2) {return node.refEllipse.dist(e1) < node.refEllipse.dist(e2); });

				size_t range = node.rIndex - node.lIndex;
				size_t midIndex = node.lIndex + (range / 2);
				node.radius = node.refEllipse.dist(points[midIndex]);

				node.isLeaf = false;
				nodes.push_back(Node(true, nodes[procIndex].lIndex, midIndex));
				nodes.push_back(Node(true, midIndex, nodes[procIndex].rIndex));

				nodes[procIndex].lIndex = nodes.size() - 2;
				nodes[procIndex].rIndex = nodes.size() - 1;
				nodeProcStack.push_back(nodes[procIndex].lIndex);
				nodeProcStack.push_back(nodes[procIndex].rIndex);
			}*/
		}

	}

	template <typename QueryEllipse, typename AccFunc>
	void rangeSearchLinear(QueryEllipse & qe, AccFunc accFunc, float range) {
        
        for (size_t i = 0; i < points.size(); ++i) {
			float dist = qe.dist(points[i]);
            if (dist < range) {
				accFunc(points[i], dist, i);
			}
		}
	}
    
    template <typename QueryEllipse, typename AccFunc>
    void rangeSearchLinearTerminated(QueryEllipse & qe, AccFunc accFunc, float range) {
        for (size_t i = 0; i < points.size(); ++i) {
            float dist = qe.dist(points[i]);
            if (dist < range) {
                if (accFunc(points[i], dist, i))
                    return;
            }
        }
    }

	template <typename QueryEllipse, typename AccFunc>
	void rangeSearch(QueryEllipse & qe, AccFunc accFunc, float range) {
		std::vector<size_t> nodesToVisit;
		nodesToVisit.push_back(0);

		while (!nodesToVisit.empty()) {
			size_t idx = nodesToVisit.back();
			Node & node = nodes[idx];
			nodesToVisit.pop_back();

			float dist = qe.dist(node.refEllipse);
			if (dist < range) {
				accFunc(points[node.refIndex], dist, node.refIndex);
			}

			if (node.radius >= 0) {
				//node has children, we need to examine them
				if (dist <= node.radius) {
					//in inner node
					if (node.lIndex>=0) nodesToVisit.push_back(node.lIndex);
					if (node.rIndex >= 0 && dist + range >= node.radius) {
						nodesToVisit.push_back(node.rIndex);
					}
				}
				else {
					//in outer node
					if (node.rIndex>=0) nodesToVisit.push_back(node.rIndex);
					if (node.lIndex>=0 && dist - range <= node.radius) {
						nodesToVisit.push_back(node.lIndex);
					}
				}
			}
			
		}

	}

	template <typename QueryEllipse, typename AccFunc>
	void rangeSearchTerminated(QueryEllipse & qe, AccFunc accFunc, float range) {
		std::vector<size_t> nodesToVisit;
		nodesToVisit.push_back(0);

		while (!nodesToVisit.empty()) {
			size_t idx = nodesToVisit.back();
			Node & node = nodes[idx];
			nodesToVisit.pop_back();

			float dist = qe.dist(node.refEllipse);
			if (dist < range) {
				if (accFunc(points[node.refIndex], dist, node.refIndex))
					return;
			}

			if (node.radius >= 0) {
				//node has children, we need to examine them
				if (dist <= node.radius) {
					//in inner node
					if (node.lIndex >= 0) nodesToVisit.push_back(node.lIndex);
					if (node.rIndex >= 0 && dist + range >= node.radius) {
						nodesToVisit.push_back(node.rIndex);
					}
				}
				else {
					//in outer node
					if (node.rIndex >= 0) nodesToVisit.push_back(node.rIndex);
					if (node.lIndex >= 0 && dist - range <= node.radius) {
						nodesToVisit.push_back(node.lIndex);
					}
				}
			}

		}

	}
	

	std::vector<Node> nodes;
	const std::vector<RefEllipse>& points;
	std::vector<std::pair<size_t, float>> pointIndices;
};
