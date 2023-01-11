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

#include <opencv2/core/core.hpp>

static double Gaussian(double x, double variance) {
    double exponent = (x * x) / (2 * variance);
    return exp(-exponent);
}
static double Gaussian(double x, double center, double variance) {
    return Gaussian(x - center, variance);
}

static cv::Vec3b Int2Color(const uint64_t _input) {
    uint64_t hash = 14695981039346656037ull;
    const unsigned char * input = (unsigned char*)&_input;
    for (int i = 0; i < 8; i++) {
        hash ^= input[i];
        hash *= 1099511628211ull;
    }
    return cv::Vec3b(hash & 255, (hash >> 8) & 255, (hash >> 16) & 255);
}

static float SquaredDistance(cv::Point2f a, cv::Point2f b) {
    return ((a.x - b.x) * (a.x - b.x)) + ((a.y - b.y) * (a.y - b.y));
}

static float Distance(cv::Point2f a, cv::Point2f b) {
    return sqrt(SquaredDistance(a, b));
}

template <typename Key, typename Item>
auto SeparateInBins(const std::vector<Item>& items,
    std::function<Key(Item)> GetKey,
    std::function<bool(const Key&, const Key&)> IsApprox)
    ->std::vector<std::vector<Item>> {

    std::vector<bool> items_sorted(items.size(), false);
    std::vector<std::vector<Item>> ret;

    for (int i = 0; i < items.size(); ++i) {
        if (items_sorted[i]) continue;
        const Key key = GetKey(items[i]);
        ret.push_back({ items[i] });
        items_sorted[i] = true;
        for (int j = i + 1; j < items.size(); ++j) {
            if (items_sorted[j]) continue;
            const Item& other = items[j];
            if (IsApprox(key, GetKey(other))) {
                ret.back().push_back(other); // copy
                items_sorted[j] = true;
            }
        }
    }
    return ret;
}

class InputParser {
    // https://stackoverflow.com/questions/865668/how-to-parse-command-line-arguments-in-c
public:
    InputParser(int& argc, char** argv) {
        for (int i = 1; i < argc; ++i) {
            tokens.push_back(std::string(argv[i]));
        }
    }
    const std::string& getCmdOption(const std::string& option) const {
        decltype(tokens)::const_iterator it;
        it = std::find(tokens.begin(), tokens.end(), option);
        if (it != tokens.end() && ++it != tokens.end()) {
            return *it;
        }
        static const std::string empty_string("");
        return empty_string;
    }
    bool cmdOptionExists(const std::string& option) const {
        return std::find(tokens.begin(), tokens.end(), option) != tokens.end();
    }
private:
    std::vector<std::string> tokens;
};
