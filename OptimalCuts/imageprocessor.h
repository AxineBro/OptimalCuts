#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include <map>
#include <set>
#include <string>
#include <limits>

struct Cut {
    cv::Point2f p_out;      // точка на внешнем контуре
    cv::Point2f p_in;       // точка на внутреннем контуре (отверстии)
    int contour_out;         // индекс внешнего контура
    int contour_in;          // индекс внутреннего контура
    float length() const { return cv::norm(p_out - p_in); }
};

class ImageProcessor {
public:
    ImageProcessor() = default;

    void process(const cv::Mat &binImage, double approxEpsilon = 5.0); // Увеличил epsilon для скорости

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

    // ОПТИМИЗИРОВАННАЯ версия поиска минимального расстояния
    Cut findMinDistanceCutOptimized(const std::vector<cv::Point>& contour1, int idx1,
                                    const std::vector<cv::Point>& contour2, int idx2);

    // Быстрый MST только с одним разрезом на отверстие
    std::vector<Cut> buildSimpleCuts(int externalContourIdx,
                                     const std::vector<int>& holeIndices);
};
