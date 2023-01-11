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

#include "MSEREllipse.h"
#include "CommonParameters.h"

#include <opencv2/highgui/highgui.hpp> // for debugging

#include <iostream>

//#define DEBUG_SHOW_EXTRACTED_RECT

float MSEREllipse::shapeWeight = 1.f;
float MSEREllipse::apperanceWeight = 1.f;

MSEREllipse::MSEREllipse(const std::vector<cv::Point> & pts, const cv::Mat& image) {
    ConstructRectFromPoints(pts);
    CopyImageInRect(image);
}

float MSEREllipse::dist(const MSEREllipse & other) {
    if(shapeWeight == 0.0f)
        return appearanceSimilarity(*this, other);
    else if(apperanceWeight == 0.0f)
        return shapeDissimilarity(*this, other);
    else
        return shapeWeight*shapeDissimilarity(*this, other) + apperanceWeight * appearanceSimilarity(*this, other);
}

void MSEREllipse::ConstructRectFromPoints(const std::vector<cv::Point>& pts) {
    using namespace cv;

    Mat datapoints = Mat((int)pts.size(), 2, CV_32F); 
    int ctr = 0;
    for (Point p : pts) {
        datapoints.at<float>(ctr, 0) = static_cast<float>(p.x);
        datapoints.at<float>(ctr, 1) = static_cast<float>(p.y);
        ++ctr;
    }

    //fit the ellipse
    Mat mean;
    reduce(datapoints, mean, 0, CV_REDUCE_AVG);
    for (int i = 0; i < pts.size(); ++i) {
        datapoints.at<float>(i, 0) -= mean.at<float>(0);
        datapoints.at<float>(i, 1) -= mean.at<float>(1);
    }
    center = Point2f(mean.at<float>(0), mean.at<float>(1));

    PCA transform(datapoints, noArray(), PCA::DATA_AS_ROW);
    Mat evecs = transform.eigenvectors;
    Mat evals = transform.eigenvalues;

    Mat covmat, tmp;
    Mat fproj = transform.project(datapoints);
    calcCovarMatrix(fproj, covmat, tmp, COVAR_ROWS | COVAR_NORMAL | COVAR_SCALE, CV_32F);

    tmp = evecs.row(0) * 3 * sqrt(covmat.at<float>(0, 0));
    u = Vec2f(tmp.at<float>(0), tmp.at<float>(1));
    tmp = evecs.row(1) * 3 * sqrt(covmat.at<float>(1, 1));
    v = Vec2f(tmp.at<float>(0), tmp.at<float>(1));

    if (norm(u) / norm(v) > 0.95 && norm(u) / norm(v) < 1.05) {
        isCircle = true;
        // set it to be axis-aligned
        float n = static_cast<float>(norm(u) + norm(v)) / 2.0f;
        u = Vec2f(n, 0);
        v = Vec2f(0, n);
    } else {
        // ellipse is symmetric on both axis, so inverting them is no problem.
        // make u point right if possible, and up otherwise
        if (u[0] < -0.05 || (abs(u[0]) < 0.05 && u[1] < -0.05)) {
            u = -u;
        }
        // same thing for v
        if (v[0] < -0.05 || (abs(v[0]) < 0.05 && v[1] < -0.05)) {
            v = -v;
        }
    }

    brect.size.width = (float)norm(u) * 2.0f;
    brect.size.height = (float)norm(v) * 2.0f;
    brect.angle = atan2(u[1], u[0]) * 180 / 3.1415926535897932384626433832795f;
    brect.center = center;
}

void MSEREllipse::CopyImageInRect(const cv::Mat& image) {
    const int size = 20;

    cv::Point2f src[] = { center, cv::Point2f(cv::Vec2f(center) + u), cv::Point2f(cv::Vec2f(center) + v) };
    cv::Point2f dst[] = { cv::Point2f(size / 2, size / 2), cv::Point2f(size, size / 2), cv::Point2f(size / 2, size) };
    cv::Mat transform = cv::getAffineTransform(src, dst);

    cv::Mat outputImage;
    cv::warpAffine(image, outputImage, transform, cv::Size(size, size));

#ifdef DEBUG_SHOW_EXTRACTED_RECT
    Mat featureImg = image.clone();
    Vec3b color(0, 0, 255); // BGR
    cv::ellipse(featureImg, brect, color);
    namedWindow("Feature", WINDOW_GUI_NORMAL);// Create a window for display.
    imshow("Feature", featureImg);

    namedWindow("normImg", WINDOW_GUI_NORMAL);
    imshow("normImg", outputImage);
    waitKey(0);
#endif

    outputImage.convertTo(normImg, CV_32F);
    normImg /= 255.0f;
    cv::Scalar mean_d = mean(normImg); // 4x doubles
    cv::Vec3f meanVal = cv::Vec3d(mean_d.val[0], mean_d.val[1], mean_d.val[2]);
    normImg = normImg - meanVal;
}

float MSEREllipse::shapeDissimilarity(const MSEREllipse & e1, const MSEREllipse & e2) {
    using namespace std;

    cv::Size2f s1 = e1.brect.size;
    cv::Size2f s2 = e2.brect.size;
    
    return shapeDissimilarity(s1, s2);

}

float MSEREllipse::shapeDissimilarity(const cv::Size2f & s1, const cv::Size2f & s2) {
	//return std::max(
	//    abs(s1.width - s2.width) / std::max(s1.width, s2.width),
	//    abs(s1.height - s2.height) / std::max(s1.height, s2.height)
	//);
    
    //chi-square distance
	auto dif = (s1 - s2);
	auto sum = (s1 + s2);
    float distance = ((dif.width*dif.width) / sum.width) + ((dif.height*dif.height) / sum.height);
    
    return distance;
}

float MSEREllipse::appearanceSimilarity(const MSEREllipse &e1, const MSEREllipse &e2) {
	return appearanceSimilarity(e1.normImg, e2.normImg);
}

float MSEREllipse::appearanceSimilarity(const cv::Mat &m1, const cv::Mat &m2) {
	//normImg are in the range [0,1]
	float score = cv::norm(m1 - m2, cv::NORM_L2);// / m1.size().area();
	return score;
}

bool MSEREllipse::CanMakeALink(const MSEREllipse &e1, const MSEREllipse &e2, int cols, int rows){
    float dist_x = abs(e1.center.x - e2.center.x);
    float dist_y = abs(e1.center.y - e2.center.y);
    if (dist_x * MIN_REPETITION > cols) return false;
    if (dist_y * MIN_REPETITION > rows) return false;
    
    return true;
}
