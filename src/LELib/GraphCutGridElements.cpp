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

#include "GraphCutGridElements.h"

#include <opencv2/highgui/highgui.hpp>

#pragma warning(push)
#pragma warning(disable:4267) // size_t to int conversion
#include <andres/graph/graph.hxx>
#include <andres/graph/multicut-lifted/greedy-additive.hxx>
#include <andres/graph/multicut-lifted/kernighan-lin.hxx>
#pragma warning(pop)

#include "Utils.h"
#include "CommonParameters.h"

using namespace andres::graph::multicut_lifted;

//Given a list of lattices that have similar transformations,
//we will group the inlier features of these lattices into
//elements of a common grid
bool GraphCutGridElements(
                          const std::vector<MSEREllipse>& ellipses,
                          const std::vector<Lattice> lattices,
                          const cv::Mat& image,
                          ElementCluster &ec) {

    
    std::vector<Node> nodes;

    //find the bounding box of all the lattices
    cv::Vec2f minPt(static_cast<float>(image.cols), static_cast<float>(image.rows));
    cv::Vec2f maxPt(0,0);
    for (size_t l = 0; l < lattices.size(); ++l) {
        if(lattices[l].minPt(0) < minPt(0))
            minPt(0) = lattices[l].minPt(0);
        if(lattices[l].minPt(1) < minPt(1))
            minPt(1) = lattices[l].minPt(1);
        if(lattices[l].maxPt(0) > maxPt(0))
            maxPt(0) = lattices[l].maxPt(0);
        if(lattices[l].maxPt(1) > maxPt(1))
            maxPt(1) = lattices[l].maxPt(1);
    }
    
    // get grid sizes
    size_t grid_n_columns = 0;
    size_t grid_n_rows = 0;
    
    //todo:find the average grid transformation
    cv::Vec2f trans1 = lattices.front().firstGridTrans;
    cv::Vec2f trans2 = lattices.front().secondGridTrans;
    
    double grid_size_x = std::max(trans1[0], trans2[0]);
    double grid_size_y = std::max(trans1[1], trans2[1]);
    
    for (size_t l = 0; l < lattices.size(); ++l) {
        
        //in case we only detected a partial grid,
        //we can check how far we can go from the first element
        //to the top-left corner of the bounding box
        int addRow = 0;
        int addCol = 0;
        unsigned long minRow = std::numeric_limits<unsigned long>::max();
        unsigned long minCol = std::numeric_limits<unsigned long>::max();
        double c = 0;//min col start
        double r = 0;//min row start
        
        const auto& elements = lattices[l].gridElements;
        
        //find the element that has the minimum row and col indices
        //to figure out how far we can go to the top left of the grid
        for (unsigned long col = 0; col < elements.size(); ++col) {
            const auto& column = elements[col];
            
            for (unsigned long row = 0; row < column.size(); ++row) {
                const auto& items = column[row];
                if (!items.empty()) {
                    if(minRow == -1)
                    {
                        minRow = row;
                        if(grid_size_y != 0)
                            r = abs(ellipses[items.front()].center.y-minPt(1))/grid_size_y;
                    }
                    else if(row < minRow)
                    {
                        minRow = row;
                        if(grid_size_y != 0)
                            r = abs(ellipses[items.front()].center.y-minPt(1))/grid_size_y;
                    }
                    
                    if(minCol == -1)
                    {
                        minCol = col;
                        if(grid_size_x != 0)
                            c = abs(ellipses[items.front()].center.x-minPt(0))/grid_size_x;
                    }
                    else if(col < minCol)
                    {
                        minCol = col;
                        if(grid_size_x != 0)
                            c = abs(ellipses[items.front()].center.x-minPt(0))/grid_size_x;
                    }
                }
            }
        }
        //printf("r:%f c:%f\n", r, c);
        if(ceil(r) - r < 0.05)
            addRow = ceil(r);
        else
            addRow = floor(r);
        addRow -= minRow;
                        
        if(ceil(c) - c < 0.05)
            addCol = ceil(c);
        else
            addCol = floor(c);
        addCol -= minCol;
                        
        //printf("Lattice %d:Additional row and col:%d, %d\n", lattices[l].latticeIndex, addRow, addCol);
                        
        if (grid_n_columns < elements.size()+addCol) {
            grid_n_columns = elements.size()+addCol;
        }
                        
        if (grid_n_rows < elements[0].size()+addRow) {
            grid_n_rows = elements[0].size()+addRow;
        }
        
        //update the elements in the lattice with new row/col indices
        for (size_t col = 0; col < elements.size(); ++col) {
            const auto& column = elements[col];
            
            for (size_t row = 0; row < column.size(); ++row) {
                const auto& items = column[row];
                if (!items.empty()) {
                    // we only take one feature per cell
                    nodes.push_back({ items.front(), l, (size_t)addRow + row, (size_t)addCol + col });
                }
            }
        }
    }

    // each ellipse is a vertex in the graph
    andres::graph::Graph<> graph;
    graph.insertVertices(nodes.size());
    std::vector<double> weights; // size = number of edges in graph
    
    for (int i = 0; i < nodes.size(); ++i) {
        const MSEREllipse& e1 = ellipses[nodes[i].idx];
        for (int j = i + 1; j < nodes.size(); ++j) {
            const MSEREllipse& e2 = ellipses[nodes[j].idx];
            
            if (nodes[i].lattice == nodes[j].lattice) {
                graph.insertEdge(i, j);
                weights.push_back(-10);
            } else if (nodes[i].row != nodes[j].row || nodes[i].col != nodes[j].col) {
                graph.insertEdge(i, j);
                weights.push_back(-1);
            } else {
                // weight scales with grid distance
                // from 1 (same position) to 0 (same as grid size)
                //double dst = 1 - Distance(e1.center, e2.center) / grid_size; // range [1, 0-]
                double dst_x = 0;
                double dst_y = 0;
                if(grid_size_x != 0)
                    dst_x = abs(e1.center.x - e2.center.x) / grid_size_x;
                if(grid_size_y != 0)
                    dst_y = abs(e1.center.y - e2.center.y) / grid_size_y;
                double dst = std::max(dst_x, dst_y); // worse one
                //if (dst > 0) {
                    graph.insertEdge(i, j);
                    weights.push_back(1 - dst);
                //}
            }
        }
    }
    assert(graph.numberOfEdges() == weights.size());
    
#ifdef VERBOSE
    printf("Graph has %zu nodes and %zu edges\n", graph.numberOfVertices(), graph.numberOfEdges());

    printf("Graph cut: computing... ");
#endif
    
    clock_t start = clock();
    size_t n_labels = grid_n_columns * grid_n_rows;
    std::vector<unsigned long> tmp_labels = greedyAdditiveEdgeContraction(graph, graph, weights, n_labels, n_labels);
    
#ifdef VERBOSE
    printf("done (%f s)\n", (clock() - start) / (double)CLOCKS_PER_SEC);
    printf("Graph cut: computing...");
#endif
    
    KernighanLinSettings settings;
    settings.introduce_new_sets = false;
    std::vector<unsigned long> vertex_labels = kernighanLin(graph, graph, weights, tmp_labels, settings);
    
#ifdef VERBOSE
    printf("done (%f s)\n", (clock() - start) / (double)CLOCKS_PER_SEC);
#endif
    
    // construct clusters vector from labels
    assert(vertex_labels.size() == nodes.size());
    size_t maxlabel = *std::max_element(vertex_labels.begin(), vertex_labels.end());

    //we will see if we have detected every possible grid element
    std::vector<std::vector<bool> > elementFound;
    int numElementFound = 0;
    elementFound.resize(grid_n_columns);
    for(int c=0; c<grid_n_columns; c++){
        elementFound[c].resize(grid_n_rows, false);
    }
    
    ec.nRows = grid_n_rows;
    ec.nCols = grid_n_columns;
    ec.firstGridTrans = trans1;
    ec.secondGridTrans = trans2;
    ec.elements.resize(maxlabel+1);
    
    for (int i = 0; i < vertex_labels.size(); ++i) {
        if(!elementFound[nodes[i].col][nodes[i].row]){
            elementFound[nodes[i].col][nodes[i].row] = true;
            numElementFound += 1;
        }
        ec.elements[vertex_labels[i]].push_back(nodes[i]);
    }
    
    //if the overall grid has too many missing elements, we will discard it
    if((float)(grid_n_rows*grid_n_columns-numElementFound) / (grid_n_rows*grid_n_columns) > MAX_ALLOWED_PERCENTAGE_OF_GRID_ELEMENTS_MISSED)
        return false;

#ifdef DEBUG_SAVE_GRID_ELEMENT_GROUPS
    cv::Mat img_copy = image.clone();
#endif
    
    std::vector<cv::Vec2f> clusterMinPts;
    std::vector<cv::Vec2f> clusterMaxPts;
    std::vector<cv::Vec2f> clusterCenterPts;
    for (int c = 0; c < ec.elements.size(); ++c) {
        cv::Vec2f clusterMinPt(image.cols, image.rows);
        cv::Vec2f clusterMaxPt(0,0);
        for (Node i : ec.elements[c]) {
            const MSEREllipse& e = ellipses[i.idx];
#ifdef DEBUG_SAVE_GRID_ELEMENT_GROUPS
            cv::ellipse(img_copy, e.brect, Int2Color(c));
#endif
            if(e.center.x < clusterMinPt(0))
                clusterMinPt(0) = e.center.x;
            if(e.center.x > clusterMaxPt(0))
                clusterMaxPt(0) = e.center.x;
            if(e.center.y < clusterMinPt(1))
                clusterMinPt(1) = e.center.y;
            if(e.center.y > clusterMaxPt(1))
                clusterMaxPt(1) = e.center.y;
        }
        ec.bboxMinPts.push_back(clusterMinPt);
        ec.bboxMaxPts.push_back(clusterMaxPt);
        ec.bboxCenterPts.push_back((clusterMinPt+clusterMaxPt)/2.0);
        
#ifdef DEBUG_SAVE_GRID_ELEMENT_GROUPS
        cv::rectangle(img_copy, cv::Point2f(clusterMinPt(0), clusterMinPt(1)), cv::Point2f(clusterMaxPt(0), clusterMaxPt(1)), Int2Color(c), 3);
#endif
    }
    
#ifdef  DEBUG_SAVE_GRID_ELEMENT_GROUPS
    static int count = 0;
    char buffer[1024];
    sprintf(buffer, "out%d.png", count++);
    cv::imwrite(buffer,img_copy);
#endif
    
    return true;
}
