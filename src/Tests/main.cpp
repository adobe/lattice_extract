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

#define CATCH_CONFIG_MAIN
#include "catch.hpp"

#include <array>
#include <vector>
#include <string>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "../LELib/CommonParameters.h"
#include "../LELib/DetectLatticeInImage.h"

#if defined(WIN32) || defined(_WIN32)
#include <filesystem>
#endif


TEST_CASE("regression test") {

    // not for osx, sorry
#if defined(WIN32) || defined(_WIN32)
    const auto& base_path = std::filesystem::current_path();
    
    const auto& success_folder = base_path / "success";
    const auto& fail_folder = base_path / "fail";

    std::array<std::pair<std::filesystem::path, bool>, 2> cases = { 
        std::make_pair(success_folder, true), 
        std::make_pair(fail_folder, false) };

    for (const auto& pair : cases) {
        std::cout << std::endl << "=== Case: " << (pair.second ? "success" : "failure") << " ===" << std::endl;
        for (const std::filesystem::directory_entry& p : std::filesystem::directory_iterator(pair.first)) {
            if (p.path().extension() == ".png" || p.path().extension() == ".jpg") {
                //std::cout << p.path().filename() << std::endl;
                std::filesystem::path working_dir = p.path().parent_path() / p.path().stem();
                std::filesystem::create_directory(working_dir);
                std::filesystem::current_path(working_dir);

                clock_t start = clock();
				std::vector<std::string> noOut;
                cv::Mat image = cv::imread(p.path().string(), CV_LOAD_IMAGE_COLOR);
				cv::Mat outputImage;
				bool success;
				#ifdef DEBUG_BULK_PATTERN_EXTRACTION_RESULTS
					success = detectLatticeInImage(image, p.path().filename().string(), outputImage, ".", noOut);
				#else
					int maxIter = -1;
					success = detectLatticeInImage(image, outputImage, maxIter);
				#endif
                 
                image.release();
                printf("%s: %f\n", p.path().filename().string().c_str(), (clock() - start) / (double)CLOCKS_PER_SEC);

                REQUIRE(success == pair.second);
            }
        }
    }
#endif
}
