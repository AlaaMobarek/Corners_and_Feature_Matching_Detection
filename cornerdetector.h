#ifndef CORNERDETECTOR_H
#define CORNERDETECTOR_H

#include <opencv2/opencv.hpp>
#include <vector>

class CornerDetector {
public:
    enum class Method { HARRIS, SHI_TOMASI };

    CornerDetector();
    ~CornerDetector();

    // تم مسح keypointScale من هنا
    std::vector<cv::KeyPoint> detect(const cv::Mat& image, Method method, float threshold, int windowSize, float sigma, float k);

    cv::Mat drawKeypoints(const cv::Mat& image, const std::vector<cv::KeyPoint>& keypoints, bool drawRich = false, cv::Scalar color = cv::Scalar(0, 0, 255));

    cv::Mat getThresholdMap() const;

private:
    cv::Mat lastThresholdMap;
};

#endif // CORNERDETECTOR_H
