#include "feature_matching.h"
#include <limits>
#include <cmath>

// ---------------------------------------------------------
// 1. Feature Matching using Sum of Squared Differences (SSD)
// ---------------------------------------------------------
std::vector<cv::DMatch> matchFeaturesSSD(const cv::Mat& desc1, const cv::Mat& desc2, float ratio_thresh) {
    std::vector<cv::DMatch> matches;

    // بنلف على كل نقطة (Descriptor) في الصورة الأولى
    for (int i = 0; i < desc1.rows; i++) {
        float minSSD = std::numeric_limits<float>::max();
        float secondMinSSD = std::numeric_limits<float>::max();
        int bestMatchIdx = -1;

        const float* d1 = desc1.ptr<float>(i); // الـ Vector الخاص بالنقطة i

        // بنقارنها بكل النقط (Descriptors) في الصورة التانية
        for (int j = 0; j < desc2.rows; j++) {
            const float* d2 = desc2.ptr<float>(j);
            float ssd = 0.0f;

            // حساب الـ SSD بين الـ 128 عنصر
            for (int k = 0; k < desc1.cols; k++) {
                float diff = d1[k] - d2[k];
                ssd += diff * diff;
            }

            // حفظ أفضل وأقرب نقطتين (عشان الـ Ratio Test بتاع SIFT)
            if (ssd < minSSD) {
                secondMinSSD = minSSD;
                minSSD = ssd;
                bestMatchIdx = j;
            } else if (ssd < secondMinSSD) {
                secondMinSSD = ssd;
            }
        }

        // تطبيق Lowe's Ratio Test لفلترة الخطوط العشوائية (Noise)
        if (minSSD < (ratio_thresh * ratio_thresh) * secondMinSSD) {
            // بنسجل إن النقطة i في الصورة الأولى اتوصلت بالنقطة bestMatchIdx في التانية
            matches.push_back(cv::DMatch(i, bestMatchIdx, minSSD));
        }
    }
    return matches;
}


// ---------------------------------------------------------
// 2. Feature Matching using Normalized Cross Correlation (NCC)
// ---------------------------------------------------------
std::vector<cv::DMatch> matchFeaturesNCC(const cv::Mat& desc1, const cv::Mat& desc2, float ratio_thresh) {
    std::vector<cv::DMatch> matches;

    for (int i = 0; i < desc1.rows; i++) {
        float maxNCC = -std::numeric_limits<float>::max();
        float secondMaxNCC = -std::numeric_limits<float>::max(); // ضفنا متغير لتاني أفضل تطابق
        int bestMatchIdx = -1;

        const float* d1 = desc1.ptr<float>(i);

        float sum_d1_sq = 0.0f;
        for(int k = 0; k < desc1.cols; k++) sum_d1_sq += d1[k] * d1[k];

        for (int j = 0; j < desc2.rows; j++) {
            const float* d2 = desc2.ptr<float>(j);

            float numerator = 0.0f;
            float sum_d2_sq = 0.0f;

            for (int k = 0; k < desc1.cols; k++) {
                numerator += d1[k] * d2[k];
                sum_d2_sq += d2[k] * d2[k];
            }

            float denominator = std::sqrt(sum_d1_sq * sum_d2_sq);
            float ncc = 0.0f;
            if (denominator != 0) {
                ncc = numerator / denominator;
            }

            // حفظ أفضل وأقرب نقطتين عشان الـ Ratio Test
            if (ncc > maxNCC) {
                secondMaxNCC = maxNCC;
                maxNCC = ncc;
                bestMatchIdx = j;
            } else if (ncc > secondMaxNCC) {
                secondMaxNCC = ncc;
            }
        }

        // ===================================================
        // تطبيق Lowe's Ratio Test على الـ NCC
        // المسافة في الـ NCC بنعبر عنها بـ (1 - قيمة الارتباط)
        // فكل ما الرقم يقرب لصفر يبقى أفضل
        // ===================================================
        float best_dist = 1.0f - maxNCC;
        float second_best_dist = 1.0f - secondMaxNCC;

        if (best_dist < ratio_thresh * second_best_dist) {
            matches.push_back(cv::DMatch(i, bestMatchIdx, best_dist));
        }
    }
    return matches;
}
