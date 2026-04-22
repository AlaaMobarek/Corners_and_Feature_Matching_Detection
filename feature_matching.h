#ifndef FEATURE_MATCHING_H
#define FEATURE_MATCHING_H

#include <opencv2/opencv.hpp>
#include <vector>

// دالة المطابقة باستخدام SSD (بنستخدم معاها Ratio Test عشان نصفي النقط الغلط)
std::vector<cv::DMatch> matchFeaturesSSD(const cv::Mat& desc1, const cv::Mat& desc2, float ratio_thresh = 0.75f);

// // دالة المطابقة باستخدام NCC (بنستخدم معاها Threshold ثابت)
// std::vector<cv::DMatch> matchFeaturesNCC(const cv::Mat& desc1, const cv::Mat& desc2, float threshold = 0.85f);
// دالة المطابقة باستخدام NCC (تم إضافة الـ Ratio Test لها)
std::vector<cv::DMatch> matchFeaturesNCC(const cv::Mat& desc1, const cv::Mat& desc2, float ratio_thresh = 0.75f);

#endif // FEATURE_MATCHING_H
