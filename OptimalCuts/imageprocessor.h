#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>

struct Cut {
    cv::Point2f p_out;
    cv::Point2f p_in;
    int contour_out;
    int contour_in;
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

    Cut findMinDistanceCutOptimized(const std::vector<cv::Point>& contour1, int idx1,
                                    const std::vector<cv::Point>& contour2, int idx2);

    std::vector<Cut> buildSimpleCuts(int externalContourIdx,
                                     const std::vector<int>& holeIndices);
};
