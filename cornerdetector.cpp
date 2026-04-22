#include "CornerDetector.h"
#include <cmath>

CornerDetector::CornerDetector() {}
CornerDetector::~CornerDetector() {}

cv::Mat CornerDetector::getThresholdMap() const {
    return lastThresholdMap;
}

// تم مسح keypointScale من تعريف الدالة هنا
std::vector<cv::KeyPoint> CornerDetector::detect(const cv::Mat& image, Method method, float threshold, int windowSize, float sigma, float k) {
    std::vector<cv::KeyPoint> keypoints;
    if (image.empty()) return keypoints;

    cv::Mat gray;
    if (image.channels() == 3) cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    else gray = image.clone();

    cv::Mat Ix, Iy;
    cv::Sobel(gray, Ix, CV_32F, 1, 0, 3);
    cv::Sobel(gray, Iy, CV_32F, 0, 1, 3);

    cv::Mat Ix2, Iy2, Ixy;
    cv::multiply(Ix, Ix, Ix2);
    cv::multiply(Iy, Iy, Iy2);
    cv::multiply(Ix, Iy, Ixy);

    if (windowSize % 2 == 0) windowSize++;

    cv::GaussianBlur(Ix2, Ix2, cv::Size(windowSize, windowSize), sigma);
    cv::GaussianBlur(Iy2, Iy2, cv::Size(windowSize, windowSize), sigma);
    cv::GaussianBlur(Ixy, Ixy, cv::Size(windowSize, windowSize), sigma);

    cv::Mat R = cv::Mat::zeros(gray.size(), CV_32F);

    if (method == Method::HARRIS) {
        cv::Mat Ix2_Iy2, Ixy_2, trace, trace2, det;
        cv::multiply(Ix2, Iy2, Ix2_Iy2);
        cv::multiply(Ixy, Ixy, Ixy_2);
        det = Ix2_Iy2 - Ixy_2;

        trace = Ix2 + Iy2;
        cv::multiply(trace, trace, trace2);

        R = det - k * trace2;
    }
    else if (method == Method::SHI_TOMASI) {
        for (int r = 0; r < gray.rows; r++) {
            for (int c = 0; c < gray.cols; c++) {
                float ix2 = Ix2.at<float>(r, c);
                float iy2 = Iy2.at<float>(r, c);
                float ixy = Ixy.at<float>(r, c);

                float diff = ix2 - iy2;
                float root = std::sqrt(diff * diff + 4.0f * ixy * ixy);
                R.at<float>(r, c) = 0.5f * (ix2 + iy2 - root);
            }
        }
    }

    cv::normalize(R, R, 0.0, 1.0, cv::NORM_MINMAX);
    R.convertTo(this->lastThresholdMap, CV_8UC1, 255.0);

    for (int r = 1; r < R.rows - 1; r++) {
        for (int c = 1; c < R.cols - 1; c++) {
            float current_response = R.at<float>(r, c);

            if (current_response > threshold) {
                bool isLocalMax = true;
                for (int i = -1; i <= 1; i++) {
                    for (int j = -1; j <= 1; j++) {
                        if (i == 0 && j == 0) continue;
                        if (R.at<float>(r + i, c + j) >= current_response) {
                            isLocalMax = false;
                            break;
                        }
                    }
                    if (!isLocalMax) break;
                }

                if (isLocalMax) {
                    cv::KeyPoint kp;
                    kp.pt = cv::Point2f(static_cast<float>(c), static_cast<float>(r));

                    // هنا التغيير: خلينا حجم الدائرة ثابت بـ 8.0 عشان الرسم
                    kp.size = 8.0f;

                    kp.response = current_response;
                    keypoints.push_back(kp);
                }
            }
        }
    }

    return keypoints;
}

cv::Mat CornerDetector::drawKeypoints(const cv::Mat& image, const std::vector<cv::KeyPoint>& keypoints, bool isThresholdMap, cv::Scalar color) {
    cv::Mat resultImg;

    if (isThresholdMap) {
        resultImg = cv::Mat::zeros(image.size(), CV_8UC3);
        for (const auto& kp : keypoints) {
            cv::circle(resultImg, kp.pt, 1, cv::Scalar(255, 255, 255), -1, cv::LINE_AA);
        }
    } else {
        if (image.channels() == 1) cv::cvtColor(image, resultImg, cv::COLOR_GRAY2BGR);
        else resultImg = image.clone();

        cv::drawKeypoints(resultImg, keypoints, resultImg, color, cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    }
    return resultImg;
}
