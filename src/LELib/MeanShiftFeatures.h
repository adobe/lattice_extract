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

#include "MSEREllipse.h"

//This function clusters the detected MSER features based on similarity
//Returns clusters as vector of indexes of ellipse
//the last argument depends on whether the distance we use is metric or not
//if the distance is metric, we use a tree for queries for speed-up
std::vector<std::vector<unsigned long>> MeanShiftFeatures(
	int& outMaxIterCount,
	const std::vector<MSEREllipse>& ellipses, const cv::Mat& image, float shapeWeight=1.0f, float appearanceWeight=1.0f, bool treeSearch=true);
