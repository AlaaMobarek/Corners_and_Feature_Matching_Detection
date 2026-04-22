#include "SiftDescriptorExtractor.h"
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

SiftDescriptorExtractor::SiftDescriptorExtractor() {}
SiftDescriptorExtractor::~SiftDescriptorExtractor() {}

// =========================================================================
// 1. بناء هرم جاوس (Gaussian Pyramid)
// =========================================================================
void SiftDescriptorExtractor::buildGaussianPyramid(const cv::Mat& baseImage, std::vector<std::vector<cv::Mat>>& pyr, int numOctaves, int numScales, float initialSigma) {
    pyr.resize(numOctaves);
    for (int o = 0; o < numOctaves; o++) {
        // كل أوكتيف بيحتاج (numScales + 3) صورة عشان نقدر نطرح ونقارن
        pyr[o].resize(numScales + 3);
    }

    cv::Mat currentImage = baseImage.clone();

    for (int o = 0; o < numOctaves; o++) {
        pyr[o][0] = currentImage.clone();
        float k = std::pow(2.0f, 1.0f / numScales);
        float sigma_prev = initialSigma;

        for (int s = 1; s < numScales + 3; s++) {
            float sigma_total = initialSigma * std::pow(k, s);
            float sigma_diff = std::sqrt(sigma_total * sigma_total - sigma_prev * sigma_prev);

            cv::GaussianBlur(pyr[o][s - 1], pyr[o][s], cv::Size(0, 0), sigma_diff, sigma_diff);
            sigma_prev = sigma_total;
        }

        // تصغير الصورة للنص عشان الأوكتيف اللي بعده (Downsampling)
        if (o < numOctaves - 1) {
            cv::resize(pyr[o][numScales], currentImage, cv::Size(currentImage.cols / 2, currentImage.rows / 2), 0, 0, cv::INTER_NEAREST);
        }
    }
}

// =========================================================================
// 2. بناء هرم الطرح (Difference of Gaussians)
// =========================================================================
void SiftDescriptorExtractor::buildDoGPyramid(const std::vector<std::vector<cv::Mat>>& gPyr, std::vector<std::vector<cv::Mat>>& dogPyr) {
    int numOctaves = gPyr.size();
    int numImages = gPyr[0].size();
    dogPyr.resize(numOctaves);

    for (int o = 0; o < numOctaves; o++) {
        dogPyr[o].resize(numImages - 1);
        for (int s = 0; s < numImages - 1; s++) {
            // بنطرح الصورة المزغللة من اللي أزغل منها
            cv::subtract(gPyr[o][s + 1], gPyr[o][s], dogPyr[o][s]);
        }
    }
}

// =========================================================================
// 3. فلترة النقط اللي واقعة على حواف (Edge Response - Hessian Matrix)
// =========================================================================
bool SiftDescriptorExtractor::passEdgeResponse(const cv::Mat& img, int r, int c, float edgeThreshold) {
    // حساب التفاضل التاني باستخدام النقط المجاورة
    float dxx = img.at<float>(r, c + 1) + img.at<float>(r, c - 1) - 2.0f * img.at<float>(r, c);
    float dyy = img.at<float>(r + 1, c) + img.at<float>(r - 1, c) - 2.0f * img.at<float>(r, c);
    float dxy = (img.at<float>(r + 1, c + 1) - img.at<float>(r + 1, c - 1) - img.at<float>(r - 1, c + 1) + img.at<float>(r - 1, c - 1)) / 4.0f;

    float tr = dxx + dyy; // Trace
    float det = dxx * dyy - dxy * dxy; // Determinant

    // لو الـ Determinant سالب، يبقى دي مش نقطة أصلاً
    if (det <= 0) return false;

    // معادلة Lowe المشهورة للحواف
    float r_th = edgeThreshold;
    if ((tr * tr) / det < ((r_th + 1.0f) * (r_th + 1.0f)) / r_th) {
        return true; // النقطة كويسة وعدت الاختبار
    }
    return false; // النقطة عبارة عن خط مستقيم ومرفوضة
}

// =========================================================================
// 4. اكتشاف القمم بين الـ 26 نقطة (Local Extrema Detection)
// =========================================================================
bool SiftDescriptorExtractor::isExtremum(const std::vector<cv::Mat>& dogOctave, int s, int r, int c) {
    float val = dogOctave[s].at<float>(r, c);
    bool isMax = true, isMin = true;

    for (int ds = -1; ds <= 1; ds++) {
        for (int dr = -1; dr <= 1; dr++) {
            for (int dc = -1; dc <= 1; dc++) {
                if (ds == 0 && dr == 0 && dc == 0) continue;
                float neighbor = dogOctave[s + ds].at<float>(r + dr, c + dc);
                if (val <= neighbor) isMax = false;
                if (val >= neighbor) isMin = false;
                if (!isMax && !isMin) return false;
            }
        }
    }
    return isMax || isMin;
}

std::vector<cv::KeyPoint> SiftDescriptorExtractor::findExtrema(const std::vector<std::vector<cv::Mat>>& dogPyr, int numOctaves, int numScales, float initialSigma, float contrastThreshold, float edgeThreshold) {
    std::vector<cv::KeyPoint> keypoints;

    for (int o = 0; o < numOctaves; o++) {
        for (int s = 1; s <= numScales; s++) {
            const cv::Mat& img = dogPyr[o][s];

            for (int r = 1; r < img.rows - 1; r++) {
                for (int c = 1; c < img.cols - 1; c++) {
                    float val = std::abs(img.at<float>(r, c));

                    // فلترة 1: التباين (Contrast) لازم يكون أقوى من الـ Threshold
                    if (val < contrastThreshold) continue;

                    // فلترة 2: هل هي الأكبر/الأصغر وسط الـ 26 جارة؟
                    if (!isExtremum(dogPyr[o], s, r, c)) continue;

                    // فلترة 3: اختبار الحواف (Hessian Matrix)
                    if (!passEdgeResponse(img, r, c, edgeThreshold)) continue;

                    // لو النقطة نجت من كل ده، نسجلها ونحسب مكانها وحجمها الفعلي
                    cv::KeyPoint kp;
                    // نضرب في 2^o عشان نرجع النقطة لإحداثيات الصورة الأصلية قبل ما تتصغر
                    kp.pt.x = c * std::pow(2.0f, o);
                    kp.pt.y = r * std::pow(2.0f, o);

                    // السكيل التلقائي اللي كنا بنحكي عنه!
                    kp.size = initialSigma * std::pow(2.0f, o + static_cast<float>(s) / numScales);

                    keypoints.push_back(kp);
                }
            }
        }
    }
    return keypoints;
}

// =========================================================================
// 5. الدالة الرئيسية اللي بتجمع الشغل ده كله وتنده من الواجهة (Detect)
// =========================================================================
std::vector<cv::KeyPoint> SiftDescriptorExtractor::detect(const cv::Mat& image, int numOctaves, int numScales, float initialSigma, float contrastThreshold, float edgeThreshold) {
    std::vector<cv::KeyPoint> keypoints;
    if (image.empty()) return keypoints;

    cv::Mat gray;
    if (image.channels() == 3) cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    else gray = image.clone();

    // لازم نحول الصورة لـ Float من 0 لـ 1 عشان العمليات الرياضية الدقيقة
    gray.convertTo(gray, CV_32F, 1.0 / 255.0);

    std::vector<std::vector<cv::Mat>> gPyr, dogPyr;
    buildGaussianPyramid(gray, gPyr, numOctaves, numScales, initialSigma);
    buildDoGPyramid(gPyr, dogPyr);

    return findExtrema(dogPyr, numOctaves, numScales, initialSigma, contrastThreshold, edgeThreshold);
}

// =========================================================================
// باقي الدوال بتاعتك (Orientation و Compute) زي ما هي بالظبط!
// =========================================================================

std::vector<cv::KeyPoint> SiftDescriptorExtractor::assignOrientation(const cv::Mat& mag, const cv::Mat& angle, const cv::KeyPoint& kp) {
    int num_bins = 36;
    std::vector<float> hist(num_bins, 0.0f);
    int radius = static_cast<int>(std::round(3.0f * 1.5f * kp.size));
    float sigma = 1.5f * kp.size;
    float exp_scale = -1.0f / (2.0f * sigma * sigma);

    // 1. بناء الـ Histogram
    for (int i = -radius; i <= radius; i++) {
        for (int j = -radius; j <= radius; j++) {
            int y = static_cast<int>(kp.pt.y + i);
            int x = static_cast<int>(kp.pt.x + j);
            if (y >= 0 && y < mag.rows && x >= 0 && x < mag.cols) {
                float m = mag.at<float>(y, x);
                float a = angle.at<float>(y, x);
                float weight = std::exp((i * i + j * j) * exp_scale);
                int bin = static_cast<int>(std::round(a * num_bins / 360.0f)) % num_bins;
                if (bin < 0) bin += num_bins;
                hist[bin] += m * weight;
            }
        }
    }

    // 2. عمل Smoothing للـ Histogram (عشان نشيل النويز)
    std::vector<float> smooth_hist(num_bins, 0.0f);
    for (int i = 0; i < num_bins; i++) {
        smooth_hist[i] = (hist[(i - 1 + num_bins) % num_bins] + hist[i] + hist[(i + 1) % num_bins]) / 3.0f;
    }

    // 3. إيجاد القمة العظمى
    float max_val = 0.0f;
    for (int i = 0; i < num_bins; i++) {
        if (smooth_hist[i] > max_val) {
            max_val = smooth_hist[i];
        }
    }

    // 4. استخراج الاتجاهات (أي قمة تتخطى 80% من العظمى) والـ Interpolation
    float threshold = 0.8f * max_val;
    std::vector<cv::KeyPoint> oriented_keypoints;

    for (int i = 0; i < num_bins; i++) {
        float val = smooth_hist[i];
        float left = smooth_hist[(i - 1 + num_bins) % num_bins];
        float right = smooth_hist[(i + 1) % num_bins];

        // نتأكد إنها قمة محلية (Local Peak) وعدت الـ Threshold
        if (val > left && val > right && val >= threshold) {
            // Parabolic Interpolation للمعادلة الدقيقة للزاوية
            float bin_offset = 0.5f * (left - right) / (left - 2.0f * val + right);
            float exact_bin = i + bin_offset;
            exact_bin = std::fmod(exact_bin + num_bins, static_cast<float>(num_bins));

            cv::KeyPoint new_kp = kp;
            new_kp.angle = exact_bin * (360.0f / num_bins);
            oriented_keypoints.push_back(new_kp);
        }
    }

    return oriented_keypoints;
}

cv::Mat SiftDescriptorExtractor::compute(const cv::Mat& image, std::vector<cv::KeyPoint>& keypoints, float sigma, int octaves) {
    if (image.empty() || keypoints.empty()) return cv::Mat();

    cv::Mat gray;
    if (image.channels() == 3) cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    else gray = image.clone();

    gray.convertTo(gray, CV_32F, 1.0 / 255.0);

    if (sigma > 0) cv::GaussianBlur(gray, gray, cv::Size(0, 0), sigma);

    cv::Mat gx, gy, mag, angle;
    cv::Sobel(gray, gx, CV_32F, 1, 0, 1);
    cv::Sobel(gray, gy, CV_32F, 0, 1, 1);
    cv::cartToPolar(gx, gy, mag, angle, true);

    std::vector<cv::KeyPoint> valid_keypoints;
    cv::Mat descriptors;
    int d = 4;
    int n = 8;

    // معالجة النقط وتحديد اتجاهاتها
    std::vector<cv::KeyPoint> oriented_keypoints;
    for (const auto& kp : keypoints) {
        std::vector<cv::KeyPoint> kps_with_angles = assignOrientation(mag, angle, kp);
        for (const auto& okp : kps_with_angles) {
            oriented_keypoints.push_back(okp);
        }
    }

    for (auto& kp : oriented_keypoints) {
        float hist_width = 3.0f * kp.size;
        int radius = static_cast<int>(std::round(hist_width * std::sqrt(2.0f) * (d + 1) * 0.5f));

        if (kp.pt.x - radius < 0 || kp.pt.x + radius >= gray.cols ||
            kp.pt.y - radius < 0 || kp.pt.y + radius >= gray.rows) continue;

        valid_keypoints.push_back(kp);

        cv::Mat desc = cv::Mat::zeros(1, d * d * n, CV_32F);
        float* desc_ptr = desc.ptr<float>(0);

        float angle_rad = (360.0f - kp.angle) * (M_PI / 180.0f);
        float cos_t = std::cos(angle_rad);
        float sin_t = std::sin(angle_rad);
        float exp_scale = -1.0f / (2.0f * (0.5f * d) * (0.5f * d));

        for (int i = -radius; i <= radius; i++) {
            for (int j = -radius; j <= radius; j++) {
                float c_rot = (j * cos_t - i * sin_t) / hist_width;
                float r_rot = (j * sin_t + i * cos_t) / hist_width;

                float r_bin = r_rot + d / 2.0f - 0.5f;
                float c_bin = c_rot + d / 2.0f - 0.5f;

                if (r_bin > -1.0f && r_bin < d && c_bin > -1.0f && c_bin < d) {
                    int img_y = static_cast<int>(kp.pt.y + i);
                    int img_x = static_cast<int>(kp.pt.x + j);

                    float m = mag.at<float>(img_y, img_x);
                    float a = angle.at<float>(img_y, img_x);

                    float a_rot = a - kp.angle;
                    if (a_rot < 0.0f) a_rot += 360.0f;
                    if (a_rot >= 360.0f) a_rot -= 360.0f;

                    float o_bin = a_rot * n / 360.0f;
                    float weight = std::exp(-(c_rot * c_rot + r_rot * r_rot) * exp_scale);
                    float weighted_mag = m * weight;

                    int r0 = static_cast<int>(std::floor(r_bin));
                    int c0 = static_cast<int>(std::floor(c_bin));
                    int o0 = static_cast<int>(std::floor(o_bin));
                    float r_frac = r_bin - r0;
                    float c_frac = c_bin - c0;
                    float o_frac = o_bin - o0;

                    for (int dr = 0; dr <= 1; dr++) {
                        int r_idx = r0 + dr;
                        if (r_idx >= 0 && r_idx < d) {
                            float v_r = weighted_mag * (dr == 0 ? (1.0f - r_frac) : r_frac);
                            for (int dc = 0; dc <= 1; dc++) {
                                int c_idx = c0 + dc;
                                if (c_idx >= 0 && c_idx < d) {
                                    float v_c = v_r * (dc == 0 ? (1.0f - c_frac) : c_frac);
                                    for (int do_bin = 0; do_bin <= 1; do_bin++) {
                                        int o_idx = (o0 + do_bin) % n;
                                        float v_o = v_c * (do_bin == 0 ? (1.0f - o_frac) : o_frac);
                                        int idx = (r_idx * d + c_idx) * n + o_idx;
                                        desc_ptr[idx] += v_o;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        cv::normalize(desc, desc, 1.0, 0, cv::NORM_L2);
        cv::threshold(desc, desc, 0.2, 0.2, cv::THRESH_TRUNC);
        cv::normalize(desc, desc, 1.0, 0, cv::NORM_L2);
        descriptors.push_back(desc);
    }

    keypoints = valid_keypoints;
    return descriptors;
}

cv::Mat SiftDescriptorExtractor::drawRichKeypoints(const cv::Mat& image, const std::vector<cv::KeyPoint>& keypoints) {
    cv::Mat output;
    if (image.channels() == 1) cv::cvtColor(image, output, cv::COLOR_GRAY2BGR);
    else output = image.clone();
    cv::drawKeypoints(output, keypoints, output, cv::Scalar(0, 255, 255), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    return output;
}
