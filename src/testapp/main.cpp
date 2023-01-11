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

#include <array>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>	

#include <iostream>
#include <stdlib.h>
#include <time.h>

#include "../LELib/CommonParameters.h"
#include "../LELib/DetectLatticeInImage.h"
#include "../LELib/Utils.h"

#include <sys/types.h>
//#include <windows.h>
//#include <ctime>
#include<chrono>
#include<vector>

#if defined(WIN32) || defined(_WIN32)
#include <filesystem> // C++17
#else
#include <dirent.h>
#endif

using namespace cv;
using namespace std;
using namespace chrono;

Mat resizeMat(Mat patternImage) {
	//Mat patternImage = Mat(imageSize.mWidth, imageSize.mHeight, CV_8UC3, CV_RGB(255, 255, 255));
	//patternImage.data = rawImageBuffer;
	if (patternImage.rows < 50 || patternImage.cols < 50) {
		float scale_parameter = 50.0 / min(patternImage.rows, patternImage.cols);
		int width = ceil(patternImage.cols * scale_parameter);
		int height = ceil(patternImage.rows * scale_parameter);
		resize(patternImage, patternImage, cv::Size(width, height), 0, 0, INTER_LINEAR);
	}
	return patternImage;
}

int main(int argc, char** argv) {
    if (argc <= 1) {
        cout << " Usage: display_image ImageToLoadAndDisplay" << endl;
        return EXIT_FAILURE;
    }
    /*std::string path = argv[1];
    InputParser input(argc, argv);
    std::string baseOutputDir = input.getCmdOption("--output_dir");
    if (baseOutputDir.empty()) baseOutputDir = "." 

	string imageName = path.substr(path.rfind(PATH_SEPARATOR) + 1, path.rfind(".") - path.rfind(PATH_SEPARATOR) - 1);
	printf("imageName:%s\n", imageName.c_str()); 
	Mat image = imread(path, CV_LOAD_IMAGE_COLOR);   // Read the file
	*/
	std::string baseOutputDir;
	bool success = false;
	auto start = std::chrono::high_resolution_clock::now();
	
	string path("./input");
	string star_path("./input/*");
	baseOutputDir = "./output/";
	vector<string> zeroFeatureFiles, noOutput, errorFiles, openFailed;
	//WIN32_FIND_DATA FindFileData;
	//HANDLE hFind = FindFirstFile(star_path.c_str(), &FindFileData);
	//if (hFind != INVALID_HANDLE_VALUE) {
	DIR* dirp = opendir(path.c_str());
	struct dirent * dp;
	while ((dp = readdir(dirp)) != NULL) 
	{
		//do {
		auto r_begin = chrono::high_resolution_clock::now();
		// string imageName(FindFileData.cFileName);
		string imageName(dp->d_name);
		cout<< "Name: " << imageName;
		if (imageName.find(".jpg") != string::npos || imageName.find(".png") != string::npos)
		{
			Mat image, basePattern;
			image = imread(path + "/" + imageName, CV_LOAD_IMAGE_COLOR);   // Read the file
			cout << imageName << endl;
			if (!image.data) {                            // Check for invalid input
				cout << "Could not open or find the image" << std::endl; //getchar();
				openFailed.push_back(imageName);
				continue;
			}
			try {
				auto r_end = chrono::high_resolution_clock::now();
				auto e_begin = chrono::high_resolution_clock::now();
				auto duration = duration_cast<microseconds>(r_end - r_begin);
				cout << "started! with Image Reading Time: " << duration.count();
				// baseOutputDir = "./Round3FP/"; 
				#ifdef DEBUG_BULK_PATTERN_EXTRACTION_RESULTS
					// detectLatticeInImage(image, imageName, baseOutputDir, noOutput);
					success = detectLatticeInImage(image, imageName, basePattern, baseOutputDir, noOutput);
				#else
					int maxIter = -1;
					success = detectLatticeInImage(image, basePattern, maxIter);
					/* image = resizeMat(image);
					char buffer[1024];
					sprintf(buffer, "%s%s%schhotiImage.png", baseOutputDir.c_str(), PATH_SEPARATOR, imageName.c_str());
					cv::imwrite(buffer, image); */
				#endif

				auto e_end = chrono::high_resolution_clock::now();
				cout << "\n Processing Time: " << duration_cast<microseconds>(e_end - e_begin).count();
				cout << "\n Total Time:  " << duration_cast<microseconds>(e_end - r_begin).count();
			}
			catch (const std::exception &e) {
				cout << "\n Some Unknown Exception occured.";
				errorFiles.push_back(imageName);
			}
			catch (...)
			{
				cout << "\nZero Features.";
				zeroFeatureFiles.push_back(imageName);
			}
		}
			//if (!(fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)) {
			//	names.push_back(fd.cFileName);
			//}
		//} while (FindNextFile(hFind, &FindFileData));
		//::FindClose(hFind);
	}
	//closedir(dirp);

#ifdef PROCESS_ALL
#if defined(WIN32) || defined(_WIN32)
    path = std::filesystem::current_path().string(); // get path from working directory
    //path = "../data"; // you can also specify the path manually
    for (const std::filesystem::directory_entry& p : std::filesystem::directory_iterator(path)) {
        if (p.path().extension() == ".png" || p.path().extension() == ".jpg") {
            cout << p.path().filename() << endl;
            filesystem::path working_dir = p.path().parent_path() / p.path().stem();
            filesystem::create_directory(working_dir);
            filesystem::current_path(working_dir);

            clock_t start = clock();
            Mat image = imread(p.path().string(), CV_LOAD_IMAGE_COLOR);
            bool success = detectLatticeInImage(image, p.path().filename().string(), baseOutputDir);
            image.release();
            printf("%f - %s\n", (clock() - start) / (double)CLOCKS_PER_SEC, success ? "success" : "fail");
        }
    }
    return EXIT_SUCCESS;
#else
    path = "/Users/ceylan/Documents/code/lattice_extract/data";
    DIR* dirp = opendir(path.c_str());
    struct dirent * dp;
    while ((dp = readdir(dirp)) != NULL) {
        string imageName(dp->d_name);
        if (imageName.find(".png") != string::npos || imageName.find(".jpg") != string::npos) {
            printf("imageName:%s\n", imageName.c_str());
            auto start = std::chrono::high_resolution_clock::now();
            Mat image;
            image = imread(path + "/" + imageName, CV_LOAD_IMAGE_COLOR);   // Read the file

            if (!image.data)                              // Check for invalid input
            {
                cout << "Could not open or find the image" << std::endl;
                return -1;
            }

			int result = 0;
            detectLatticeInImage(image, imageName, baseOutputDir, result);
            image.release();
            auto finish = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed = finish - start;
            printf("Total time: %f s\n", elapsed);
        }
    }
    closedir(dirp);
    return EXIT_SUCCESS;
#endif
#else
	// Removed tbegin, Only need start
    //bool success = detectLatticeInImage(image, imageName, baseOutputDir, noOutput);

	//printf("\nComplete Time: %.7fs", (clock() - tbegin) / CLOCKS_PER_SEC);
	// detectLatticeInImage(image, imageName, baseOutputDir);
	cout << "Zero Feature Files: \n";
	for (string &mystr : zeroFeatureFiles) {
		cout << endl << mystr;
	}
	cout << "No Output Files: \n";
	for (string &mystr : noOutput) {
		cout << endl << mystr;
	}
	cout << "Errored Files: \n";
	for (string &mystr : errorFiles) {
		cout << endl << mystr;
	}
	cout << "Open Failed for Files: \n";
	for (string &mystr : openFailed) {
		cout << endl << mystr;
	}

    auto finish = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = finish - start;
    printf("Total time: %f s\n", elapsed);

	//auto end = std::chrono::high_resolution_clock::now();
	//auto duration = chrono::duration_cast<microseconds>(end - start);
	//cout << "Total Exection: " << duration.count() << endl;

	getchar(); getchar();
    return success ? EXIT_SUCCESS : EXIT_FAILURE;
#endif 
}

