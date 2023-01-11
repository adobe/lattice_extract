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
#include <algorithm> // for std::max

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc.hpp>

struct MSEREllipse {
	cv::Vec2f u, v;
	cv::Point2f center;
	cv::RotatedRect brect;
	cv::Mat normImg;
	bool isCircle;

	MSEREllipse(const std::vector<cv::Point> & pts, const cv::Mat& image);
	MSEREllipse(){}

    static float shapeDissimilarity(const MSEREllipse & e1, const MSEREllipse & e2);

	static float shapeDissimilarity(const cv::Size2f & s1, const cv::Size2f & s2);
    
    static float appearanceSimilarity(const MSEREllipse &e1, const MSEREllipse &e2);
	static float appearanceSimilarity(const cv::Mat &m1, const cv::Mat &m2);

	float dist(const MSEREllipse & other);
    
    static bool CanMakeALink(const MSEREllipse& e1, const MSEREllipse& e2, int cols, int rows);

public:
	static float shapeWeight;
    static float apperanceWeight;

private:
    void ConstructRectFromPoints(const std::vector<cv::Point>& pts);
    void CopyImageInRect(const cv::Mat& image);
};
