#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QLabel>
#include <QPushButton>
#include <QSlider>
#include <QGroupBox>
#include <QRadioButton>
#include <QComboBox>
#include <QStackedWidget>
#include <opencv2/opencv.hpp>
#include <vector>

#include "CornerDetector.h"
#include "SiftDescriptorExtractor.h"

class MainWindow : public QMainWindow {
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

private slots:
    void loadMainImage();
    void onApplySIFTOrMatch();
    void applyHarris();
    void applyLambda();

private:
    void setupUI();
    void applyStyle();
    void displayImage(const cv::Mat& img, QLabel* label);

    // Layout Elements
    QPushButton *btnLoadMain;
    QLabel *originalImageLabel;
    QLabel *outputImageLabel;
    QLabel *timeLabel;

    // Menu and Dynamic Parameters UI
    QComboBox *modeSelector;
    QStackedWidget *paramsStackedWidget;
    QPushButton *btnApplyMain;

    // --- PAGE 0: Extract Features (Harris & Lambda) ---
    // Harris Elements
    QSlider *harrisThreshSlider;  QLabel *harrisThreshValLbl;
    QSlider *harrisSigmaSlider;   QLabel *harrisSigmaValLbl;
    QSlider *harrisWinSizeSlider; QLabel *harrisWinSizeValLbl;
    QSlider *harrisKSlider;       QLabel *harrisKValLbl;
    QRadioButton *radioHarrisFinal;
    QRadioButton *radioHarrisThreshold;
    QPushButton *btnApplyHarris;

    // Lambda Elements
    QSlider *lambdaThreshSlider;  QLabel *lambdaThreshValLbl;
    QSlider *lambdaSigmaSlider;   QLabel *lambdaSigmaValLbl;
    QSlider *lambdaWinSizeSlider; QLabel *lambdaWinSizeValLbl;
    QRadioButton *radioLambdaFinal;
    QRadioButton *radioLambdaThreshold;
    QPushButton *btnApplyLambda;

    // --- PAGE 1: SIFT Descriptors (DoG) ---
    QSlider *siftOctavesSlider;        QLabel *siftOctavesValLbl;
    QSlider *siftScalesSlider;         QLabel *siftScalesValLbl;
    QSlider *siftSigmaSlider;          QLabel *siftSigmaValLbl;
    QSlider *siftContrastThreshSlider; QLabel *siftContrastThreshValLbl;
    QSlider *siftEdgeThreshSlider;     QLabel *siftEdgeThreshValLbl;

    // Page 2: Feature Matching
    cv::Mat templateImage;
    QPushButton *btnLoadTemplate;
    QPushButton *btnApplyMatch;
    QRadioButton *radioSSD;
    QRadioButton *radioNCC;

    void loadTemplateImage();

    // Sliders for Feature Matching Ratio
    QSlider *ssdRatioSlider;
    QLabel *ssdRatioValLbl;
    QSlider *nccRatioSlider;
    QLabel *nccRatioValLbl;

    QLabel *inputLbl;
    QLabel *outputLbl;
    // Core Data State
    cv::Mat currentImage;
    std::vector<cv::KeyPoint> currentKeypoints;

    CornerDetector detector;
    SiftDescriptorExtractor siftExtractor;
};

#endif // MAINWINDOW_H
