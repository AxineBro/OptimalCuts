#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include <map>
#include <set>
#include <string>
#include <limits>

struct Cut {
    cv::Point2f p_out;      // Точка на внешнем контуре
    cv::Point2f p_in;       // Точка на внутреннем контуре (отверстии)
    int contour_out;        // Индекс внешнего контура
    int contour_in;         // Индекс внутреннего контура
    float length() const { return cv::norm(p_out - p_in); }
};

class ImageProcessor {
public:
    ImageProcessor() = default;

    void process(const cv::Mat &binImage, double thresholdValue = 127.0, double approxEpsilon = 0.0);

    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    std::vector<Cut> cuts;

    std::vector<std::vector<cv::Point2f>> mergedContours();
    float totalCutsLength() const;
    std::string getInfoString() const;

private:
    void findContoursAndHierarchy(const cv::Mat &bin);
    void approximateContours(double epsilon);
    void computeOptimalCuts();

    // Оптимизированная версия поиска минимального расстояния
    Cut findMinDistanceCutOptimized(const std::vector<cv::Point>& contour1, int idx1,
                                    const std::vector<cv::Point>& contour2, int idx2);

    // Быстрый MST только с одним разрезом на отверстие
    std::vector<Cut> buildSimpleCuts(int externalContourIdx,
                                     const std::vector<int>& holeIndices);
};
