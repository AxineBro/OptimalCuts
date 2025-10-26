#include "imageprocessor.h"
#include <limits>
#include <sstream>
#include <algorithm>
#include <iostream>
#include <queue>

using namespace std;

void ImageProcessor::process(const cv::Mat &binImage, double thresholdValue, double approxEpsilon) {
    contours.clear();
    hierarchy.clear();
    cuts.clear();

    if (binImage.empty() || binImage.type() != CV_8UC1) {
        cerr << "[ImageProcessor] process: входное изображение пустое или не CV_8UC1\n";
        return;
    }

    // Простая бинаризация с заданным порогом
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

    // Используем RETR_TREE для полной иерархии
    cv::findContours(tmp, rawContours, hier, cv::RETR_TREE, cv::CHAIN_APPROX_NONE); // CHAIN_APPROX_NONE для точных границ
    contours = move(rawContours);
    hierarchy = move(hier);
    cerr << "[ImageProcessor] findContours: найдено " << contours.size() << " контуров\n";
}

void ImageProcessor::approximateContours(double epsilon) {
    // Отключаем аппроксимацию или делаем минимальной, если epsilon > 0
    if (epsilon <= 0.0) return;

    for (auto &c : contours) {
        if (c.size() < 10) continue; // Не аппроксимируем слишком маленькие контуры

        double perimeter = cv::arcLength(c, true);
        double approxEpsilon = max(0.1, epsilon * perimeter / 5000.0); // Минимальная аппроксимация
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

    // Проходим по всем контурам и находим внешние контуры (те, у которых нет родителя)
    for (size_t i = 0; i < contours.size(); ++i) {
        if (hierarchy[i][3] == -1) { // Внешний контур
            vector<int> holeIndices;
            int firstChild = hierarchy[i][2]; // Индекс первого дочернего элемента

            if (firstChild != -1) {
                int currentHole = firstChild;
                while (currentHole != -1) {
                    if (!contours[currentHole].empty()) {
                        holeIndices.push_back(currentHole);
                    }
                    currentHole = hierarchy[currentHole][0]; // Следующий сосед
                }

                if (!holeIndices.empty()) {
                    vector<Cut> contourCuts = buildSimpleCuts((int)i, holeIndices);
                    cuts.insert(cuts.end(), contourCuts.begin(), contourCuts.end());
                }
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

    int step1 = contour1.size() > 100 ? max(1, (int)contour1.size() / 50) : 1;
    int step2 = contour2.size() > 100 ? max(1, (int)contour2.size() / 50) : 1;

    int best_i = 0, best_j = 0;

    for (size_t i = 0; i < contour1.size(); i += step1) {
        cv::Point2f p1 = cv::Point2f(contour1[i]);
        for (size_t j = 0; j < contour2.size(); j += step2) {
            cv::Point2f p2 = cv::Point2f(contour2[j]);
            double dist = cv::norm(p1 - p2);
            if (dist < minDist) {
                minDist = dist;
                bestCut.p_out = p1;
                bestCut.p_in = p2;
                best_i = (int)i;
                best_j = (int)j;
            }
        }
    }

    int refineRadius = 5;
    int start_i = max(0, best_i - refineRadius);
    int end_i = min((int)contour1.size(), best_i + refineRadius + 1);

    int start_j = max(0, best_j - refineRadius);
    int end_j = min((int)contour2.size(), best_j + refineRadius + 1);

    for (int i = start_i; i < end_i; ++i) {
        cv::Point2f p1 = cv::Point2f(contour1[i]);
        for (int j = start_j; j < end_j; ++j) {
            cv::Point2f p2 = cv::Point2f(contour2[j]);
            double dist = cv::norm(p1 - p2);
            if (dist < minDist) {
                minDist = dist;
                bestCut.p_out = p1;
                bestCut.p_in = p2;
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
        if (hierarchy[i][3] != -1) continue; // Пропускаем внутренние

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
