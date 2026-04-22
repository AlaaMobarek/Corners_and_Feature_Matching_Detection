#ifndef SIFTDESCRIPTOREXTRACTOR_H
#define SIFTDESCRIPTOREXTRACTOR_H

#include <opencv2/opencv.hpp>
#include <vector>

class SiftDescriptorExtractor {
public:
    SiftDescriptorExtractor();
    ~SiftDescriptorExtractor();

    // الدالة الجديدة لاكتشاف النقط بطريقة DoG
    std::vector<cv::KeyPoint> detect(const cv::Mat& image, int numOctaves = 4, int numScales = 3, float initialSigma = 1.6f, float contrastThreshold = 0.04f, float edgeThreshold = 10.0f);

    // دالة البصمة الأصلية بتاعتك زي ما هي
    cv::Mat compute(const cv::Mat& image, std::vector<cv::KeyPoint>& keypoints, float sigma = 1.6f, int octaves = 4);

    cv::Mat drawRichKeypoints(const cv::Mat& image, const std::vector<cv::KeyPoint>& keypoints);

private:
    std::vector<cv::KeyPoint> assignOrientation(const cv::Mat& mag, const cv::Mat& angle, const cv::KeyPoint& kp);

    // دوال DoG المساعدة (Private لأن المستخدم مش محتاج يشوفها)
    void buildGaussianPyramid(const cv::Mat& baseImage, std::vector<std::vector<cv::Mat>>& pyr, int numOctaves, int numScales, float initialSigma);
    void buildDoGPyramid(const std::vector<std::vector<cv::Mat>>& gPyr, std::vector<std::vector<cv::Mat>>& dogPyr);
    std::vector<cv::KeyPoint> findExtrema(const std::vector<std::vector<cv::Mat>>& dogPyr, int numOctaves, int numScales, float initialSigma, float contrastThreshold, float edgeThreshold);

    bool isExtremum(const std::vector<cv::Mat>& dogOctave, int s, int r, int c);
    bool passEdgeResponse(const cv::Mat& img, int r, int c, float edgeThreshold);
};

#endif // SIFTDESCRIPTOREXTRACTOR_H
