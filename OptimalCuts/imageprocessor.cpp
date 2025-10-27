#include "imageprocessor.h"
#include <limits>
#include <sstream>
#include <algorithm>
#include <iostream>
#include <opencv2/flann.hpp>

using namespace std;

void ImageProcessor::process(const cv::Mat &binImage, double thresholdValue, double approxEpsilon) {
    contours.clear();
    hierarchy.clear();
    cuts.clear();

    if (binImage.empty() || binImage.type() != CV_8UC1) {
        cerr << "[ImageProcessor] process: входное изображение пустое или не CV_8UC1\n";
        return;
    }

    cv::Mat bin;
    cv::threshold(binImage, bin, thresholdValue, 255, cv::THRESH_BINARY);

    findContoursAndHierarchy(bin);
    if (approxEpsilon > 0.0) approximateContours(approxEpsilon);
    computeOptimalCuts();
}

void ImageProcessor::findContoursAndHierarchy(const cv::Mat &bin) {
    cv::Mat tmp = bin.clone();
    vector<vector<cv::Point>> rawContours;
    vector<cv::Vec4i> hier;

    cv::findContours(tmp, rawContours, hier, cv::RETR_TREE, cv::CHAIN_APPROX_NONE);
    contours = move(rawContours);
    hierarchy = move(hier);
    cerr << "[ImageProcessor] findContours: найдено " << contours.size() << " контуров\n";
}

void ImageProcessor::approximateContours(double epsilon) {
    if (epsilon <= 0.0) return;

    for (auto &c : contours) {
        if (c.size() < 10) continue;

        double perimeter = cv::arcLength(c, true);
        double approxEpsilon = max(0.1, epsilon * perimeter / 5000.0);
        vector<cv::Point> approx;
        cv::approxPolyDP(c, approx, approxEpsilon, true);
        if (approx.size() >= 3) {
            c = move(approx);
        }
    }
}

void ImageProcessor::computeOptimalCuts() {
    if (contours.empty() || hierarchy.empty()) return;

    cuts.clear();

    for (size_t i = 0; i < contours.size(); ++i) {
        if (contours[i].empty()) continue;

        double signedArea = cv::contourArea(contours[i], true);
        if (signedArea <= 0) continue;

        vector<int> holeIndices;
        int firstChild = hierarchy[i][2];

        if (firstChild != -1) {
            int currentHole = firstChild;
            while (currentHole != -1) {
                if (!contours[currentHole].empty() && cv::contourArea(contours[currentHole], true) < 0) {
                    holeIndices.push_back(currentHole);
                }
                currentHole = hierarchy[currentHole][0];
            }

            if (!holeIndices.empty()) {
                vector<Cut> contourCuts = buildSimpleCuts((int)i, holeIndices);
                cuts.insert(cuts.end(), contourCuts.begin(), contourCuts.end());
            }
        }
    }

    cerr << "[ImageProcessor] computeOptimalCuts: произведено " << cuts.size() << " разрезов\n";
}

Cut ImageProcessor::findMinDistanceCutOptimized(const vector<cv::Point>& contour1, int idx1,
                                                const vector<cv::Point>& contour2, int idx2) {
    Cut bestCut;
    bestCut.contour_out = idx1;
    bestCut.contour_in = idx2;
    double minDist = numeric_limits<double>::max();

    if (contour1.empty() || contour2.empty()) return bestCut;

    const vector<cv::Point>* buildContour = &contour2;
    const vector<cv::Point>* queryContour = &contour1;
    bool outIsQuery = true;

    if (contour1.size() > contour2.size()) {
        buildContour = &contour1;
        queryContour = &contour2;
        outIsQuery = false;
    }

    cv::Mat pointsBuild(static_cast<int>(buildContour->size()), 2, CV_32F);
    for (size_t j = 0; j < buildContour->size(); ++j) {
        pointsBuild.at<float>(static_cast<int>(j), 0) = (*buildContour)[j].x;
        pointsBuild.at<float>(static_cast<int>(j), 1) = (*buildContour)[j].y;
    }

    cv::flann::Index kdTree(pointsBuild, cv::flann::KDTreeIndexParams(4));

    for (const auto& pt : *queryContour) {
        cv::Point2f pQuery = cv::Point2f(pt);

        cv::Mat query(1, 2, CV_32F);
        query.at<float>(0, 0) = pQuery.x;
        query.at<float>(0, 1) = pQuery.y;

        std::vector<int> indices(1);
        std::vector<float> dists(1);

        kdTree.knnSearch(query, indices, dists, 1);

        int buildIdx = indices[0];
        cv::Point2f pBuild = cv::Point2f((*buildContour)[buildIdx]);
        double dist = cv::norm(pQuery - pBuild);

        if (dist < minDist) {
            minDist = dist;
            if (outIsQuery) {
                bestCut.p_out = pQuery;
                bestCut.p_in = pBuild;
            } else {
                bestCut.p_out = pBuild;
                bestCut.p_in = pQuery;
            }
        }
    }

    return bestCut;
}

vector<Cut> ImageProcessor::buildSimpleCuts(int externalContourIdx,
                                            const vector<int>& holeIndices) {
    vector<Cut> simpleCuts;

    for (int holeIdx : holeIndices) {
        if (contours[externalContourIdx].empty() || contours[holeIdx].empty()) {
            continue;
        }

        Cut cut = findMinDistanceCutOptimized(contours[externalContourIdx], externalContourIdx,
                                              contours[holeIdx], holeIdx);
        simpleCuts.push_back(cut);
    }

    return simpleCuts;
}

vector<vector<cv::Point2f>> ImageProcessor::mergedContours() {
    vector<vector<cv::Point2f>> outputs;
    if (contours.empty()) return outputs;

    map<int, vector<Cut>> cutsByExternal;
    for (const auto& cut : cuts) {
        cutsByExternal[cut.contour_out].push_back(cut);
    }

    for (size_t i = 0; i < contours.size(); ++i) {
        if (contours[i].empty()) continue;

        double signedArea = cv::contourArea(contours[i], true);
        if (signedArea <= 0) continue;

        vector<cv::Point2f> mergedContour;
        for (const auto& p : contours[i]) {
            mergedContour.emplace_back(cv::Point2f(p));
        }

        auto it = cutsByExternal.find((int)i);
        if (it != cutsByExternal.end()) {
            for (const auto& cut : it->second) {
                mergedContour.push_back(cut.p_out);
                mergedContour.push_back(cut.p_in);

                const auto& hole = contours[cut.contour_in];
                for (const auto& p : hole) {
                    mergedContour.emplace_back(cv::Point2f(p));
                }

                mergedContour.push_back(cut.p_in);
                mergedContour.push_back(cut.p_out);
            }
        }

        outputs.push_back(mergedContour);
    }

    return outputs;
}

float ImageProcessor::totalCutsLength() const {
    float sum = 0.0f;
    for (const auto& cut : cuts) sum += cut.length();
    return sum;
}

string ImageProcessor::getInfoString() const {
    stringstream ss;
    ss << "Найдено контуров: " << contours.size() << "\n";

    int external = 0;
    int holes = 0;
    for (const auto& h : hierarchy) {
        if (h[3] == -1) external++;
        else holes++;
    }

    ss << "Внешние контуры: " << external << "\n";
    ss << "Отверстия: " << holes << "\n";
    ss << "Сделано разрезов: " << cuts.size() << "\n";
    ss << "Общая длина разрезов: " << totalCutsLength() << "\n";

    return ss.str();
}
