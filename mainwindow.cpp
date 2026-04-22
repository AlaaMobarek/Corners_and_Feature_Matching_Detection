#include "mainwindow.h"
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QFileDialog>
#include <QApplication>
#include <QSizePolicy>
#include <QMessageBox>
#include <QDebug>
#include "feature_matching.h"

MainWindow::MainWindow(QWidget *parent) : QMainWindow(parent) {
    setupUI();
    applyStyle();
    resize(1300, 800);
    setWindowTitle("Images Feature Extractor");
}

MainWindow::~MainWindow() {}

// 2. تم إضافة دالة تحميل القالب
void MainWindow::loadTemplateImage() {
    QString fileName = QFileDialog::getOpenFileName(this, "Select Image 2", "");
    if (!fileName.isEmpty()) {
        templateImage = cv::imread(fileName.toStdString());

        outputImageLabel->show(); // نرجع نظهر المربع اليمين
        displayImage(templateImage, outputImageLabel); // نعرض الصورة التانية في اليمين

        // التعديل الأهم: نرجع نعرض الصورة الأولى في الشمال عشان نمسح صورة "التطابق المدمجة"
        if (!currentImage.empty()) {
            displayImage(currentImage, originalImageLabel);
        }

        timeLabel->setText("Image 2 Loaded. Now click Apply Matching.");
    }
}

void MainWindow::setupUI() {
    QWidget *centralWidget = new QWidget(this);
    setCentralWidget(centralWidget);

    QVBoxLayout *mainLayout = new QVBoxLayout(centralWidget);
    mainLayout->setContentsMargins(15, 15, 15, 15);

    // Top Bar
    QHBoxLayout *topBarLayout = new QHBoxLayout();
    QLabel *appTitle = new QLabel("Image's Feature", this);
    appTitle->setObjectName("appTitle");
    appTitle->setAlignment(Qt::AlignCenter);

    btnLoadMain = new QPushButton("Load Image", this);
    btnLoadMain->setFixedWidth(150);

    topBarLayout->addWidget(appTitle, 1);
    topBarLayout->addWidget(btnLoadMain);
    mainLayout->addLayout(topBarLayout);
    mainLayout->addSpacing(10);

    // Content Layout
    QHBoxLayout *contentLayout = new QHBoxLayout();
    QVBoxLayout *leftLayout = new QVBoxLayout();
    leftLayout->setAlignment(Qt::AlignTop);
    leftLayout->setContentsMargins(0, 0, 15, 0);

    // Dropdown Menu
    QLabel *lblMenu = new QLabel("Choose The Mode:");
    lblMenu->setObjectName("boldLabel");
    modeSelector = new QComboBox();
    modeSelector->addItem("Extract The Unique Features");
    modeSelector->addItem("Generate Feature Descriptors (SIFT)");
    modeSelector->addItem("Feature Matching");
    modeSelector->setMinimumHeight(40);

    leftLayout->addWidget(lblMenu);
    leftLayout->addWidget(modeSelector);
    leftLayout->addSpacing(10);

    paramsStackedWidget = new QStackedWidget();

    // --------- PAGE 0: Feature Extraction (Harris & Lambda) ---------
    QWidget *pageExtract = new QWidget();
    QVBoxLayout *layoutExtract = new QVBoxLayout(pageExtract);
    layoutExtract->setContentsMargins(0,0,0,0);

    // --- 1. Harris Group ---
    QGroupBox *groupHarris = new QGroupBox("Harris Parameters");
    QVBoxLayout *harrisVBox = new QVBoxLayout(groupHarris);

    harrisThreshSlider = new QSlider(Qt::Horizontal); harrisThreshSlider->setRange(1, 100); harrisThreshSlider->setValue(25);
    harrisThreshValLbl = new QLabel("0.25");
    harrisSigmaSlider = new QSlider(Qt::Horizontal); harrisSigmaSlider->setRange(1, 100); harrisSigmaSlider->setValue(8);
    harrisSigmaValLbl = new QLabel("0.8");
    harrisWinSizeSlider = new QSlider(Qt::Horizontal); harrisWinSizeSlider->setRange(1, 15); harrisWinSizeSlider->setValue(2);
    harrisWinSizeValLbl = new QLabel("5");
    harrisKSlider = new QSlider(Qt::Horizontal); harrisKSlider->setRange(1, 10); harrisKSlider->setValue(4);
    harrisKValLbl = new QLabel("0.04");

    harrisVBox->addWidget(new QLabel("Threshold:"));
    QHBoxLayout *hT = new QHBoxLayout(); hT->addWidget(harrisThreshSlider); hT->addWidget(harrisThreshValLbl); harrisVBox->addLayout(hT);
    harrisVBox->addWidget(new QLabel("σ (Gaussian Window):"));
    QHBoxLayout *hS = new QHBoxLayout(); hS->addWidget(harrisSigmaSlider); hS->addWidget(harrisSigmaValLbl); harrisVBox->addLayout(hS);
    harrisVBox->addWidget(new QLabel("Window Size:"));
    QHBoxLayout *hW = new QHBoxLayout(); hW->addWidget(harrisWinSizeSlider); hW->addWidget(harrisWinSizeValLbl); harrisVBox->addLayout(hW);
    harrisVBox->addWidget(new QLabel("k Parameter:"));
    QHBoxLayout *hK = new QHBoxLayout(); hK->addWidget(harrisKSlider); hK->addWidget(harrisKValLbl); harrisVBox->addLayout(hK);

    radioHarrisFinal = new QRadioButton("Final Result"); radioHarrisFinal->setChecked(true);
    radioHarrisThreshold = new QRadioButton("Threshold Map");
    harrisVBox->addWidget(radioHarrisFinal); harrisVBox->addWidget(radioHarrisThreshold);

    btnApplyHarris = new QPushButton("Apply Harris", this);
    harrisVBox->addWidget(btnApplyHarris);
    layoutExtract->addWidget(groupHarris);

    // --- 2. Lambda (A-) Group ---
    QGroupBox *groupLambda = new QGroupBox("Shi-Tomasi Parameters");
    QVBoxLayout *lambdaVBox = new QVBoxLayout(groupLambda);

    lambdaThreshSlider = new QSlider(Qt::Horizontal); lambdaThreshSlider->setRange(1, 100); lambdaThreshSlider->setValue(10);
    lambdaThreshValLbl = new QLabel("0.10");
    lambdaSigmaSlider = new QSlider(Qt::Horizontal); lambdaSigmaSlider->setRange(1, 100); lambdaSigmaSlider->setValue(5);
    lambdaSigmaValLbl = new QLabel("0.5");
    lambdaWinSizeSlider = new QSlider(Qt::Horizontal); lambdaWinSizeSlider->setRange(1, 15); lambdaWinSizeSlider->setValue(1);
    lambdaWinSizeValLbl = new QLabel("3");

    lambdaVBox->addWidget(new QLabel("Threshold:"));
    QHBoxLayout *lT = new QHBoxLayout(); lT->addWidget(lambdaThreshSlider); lT->addWidget(lambdaThreshValLbl); lambdaVBox->addLayout(lT);
    lambdaVBox->addWidget(new QLabel("σ (Gaussian Window):"));
    QHBoxLayout *lS = new QHBoxLayout(); lS->addWidget(lambdaSigmaSlider); lS->addWidget(lambdaSigmaValLbl); lambdaVBox->addLayout(lS);
    lambdaVBox->addWidget(new QLabel("Window Size:"));
    QHBoxLayout *lW = new QHBoxLayout(); lW->addWidget(lambdaWinSizeSlider); lW->addWidget(lambdaWinSizeValLbl); lambdaVBox->addLayout(lW);

    radioLambdaFinal = new QRadioButton("Final Result"); radioLambdaFinal->setChecked(true);
    radioLambdaThreshold = new QRadioButton("Threshold Map");
    lambdaVBox->addWidget(radioLambdaFinal); lambdaVBox->addWidget(radioLambdaThreshold);

    btnApplyLambda = new QPushButton("Apply A- (Lambda)", this);
    lambdaVBox->addWidget(btnApplyLambda);
    layoutExtract->addWidget(groupLambda);
    layoutExtract->addStretch();

    paramsStackedWidget->addWidget(pageExtract);

    // --------- PAGE 1: SIFT Parameters (Pure DoG) ---------
    QWidget *pageSift = new QWidget();
    QVBoxLayout *layoutSift = new QVBoxLayout(pageSift);
    layoutSift->setContentsMargins(0,0,0,0);

    QGroupBox *groupSift = new QGroupBox("SIFT Parameters (DoG)");
    QVBoxLayout *siftVBox = new QVBoxLayout(groupSift);

    siftOctavesSlider = new QSlider(Qt::Horizontal); siftOctavesSlider->setRange(1, 8); siftOctavesSlider->setValue(4);
    siftOctavesValLbl = new QLabel("4");
    siftScalesSlider = new QSlider(Qt::Horizontal); siftScalesSlider->setRange(1, 5); siftScalesSlider->setValue(3);
    siftScalesValLbl = new QLabel("3");
    siftSigmaSlider = new QSlider(Qt::Horizontal); siftSigmaSlider->setRange(5, 30); siftSigmaSlider->setValue(16);
    siftSigmaValLbl = new QLabel("1.6");
    siftContrastThreshSlider = new QSlider(Qt::Horizontal); siftContrastThreshSlider->setRange(1, 20); siftContrastThreshSlider->setValue(4);
    siftContrastThreshValLbl = new QLabel("0.04");
    siftEdgeThreshSlider = new QSlider(Qt::Horizontal); siftEdgeThreshSlider->setRange(1, 50); siftEdgeThreshSlider->setValue(10);
    siftEdgeThreshValLbl = new QLabel("10.0");

    siftVBox->addWidget(new QLabel("Number of Octaves:"));
    QHBoxLayout *ssO = new QHBoxLayout(); ssO->addWidget(siftOctavesSlider); ssO->addWidget(siftOctavesValLbl); siftVBox->addLayout(ssO);

    siftVBox->addWidget(new QLabel("Scales per Octave:"));
    QHBoxLayout *ssS = new QHBoxLayout(); ssS->addWidget(siftScalesSlider); ssS->addWidget(siftScalesValLbl); siftVBox->addLayout(ssS);

    siftVBox->addWidget(new QLabel("Initial Blur (Sigma):"));
    QHBoxLayout *ssL = new QHBoxLayout(); ssL->addWidget(siftSigmaSlider); ssL->addWidget(siftSigmaValLbl); siftVBox->addLayout(ssL);

    siftVBox->addWidget(new QLabel("Contrast Threshold:"));
    QHBoxLayout *ssC = new QHBoxLayout(); ssC->addWidget(siftContrastThreshSlider); ssC->addWidget(siftContrastThreshValLbl); siftVBox->addLayout(ssC);

    siftVBox->addWidget(new QLabel("Edge Threshold:"));
    QHBoxLayout *ssE = new QHBoxLayout(); ssE->addWidget(siftEdgeThreshSlider); ssE->addWidget(siftEdgeThreshValLbl); siftVBox->addLayout(ssE);

    layoutSift->addWidget(groupSift);
    layoutSift->addSpacing(10);

    btnApplyMain = new QPushButton("Apply Pure SIFT", this);
    btnApplyMain->setMinimumHeight(45);
    layoutSift->addWidget(btnApplyMain);
    layoutSift->addStretch();

    paramsStackedWidget->addWidget(pageSift);

    // --------- PAGE 2: Feature Matching ---------
    QWidget *pageMatch = new QWidget();
    QVBoxLayout *layoutMatch = new QVBoxLayout(pageMatch);
    layoutMatch->setContentsMargins(0,0,0,0);

    QGroupBox *groupMatch = new QGroupBox("Matching Methods");
    QVBoxLayout *matchVBox = new QVBoxLayout(groupMatch);

    btnLoadTemplate = new QPushButton("Load Second Image", this);
    btnLoadTemplate->setMinimumHeight(35);
    matchVBox->addWidget(btnLoadTemplate);

    matchVBox->addSpacing(10);

    radioSSD = new QRadioButton("SSD Method");
    radioNCC = new QRadioButton("NCC Method");
    radioSSD->setChecked(true); // Default
    matchVBox->addWidget(radioSSD);
    matchVBox->addWidget(radioNCC);

    layoutMatch->addWidget(groupMatch);
    layoutMatch->addSpacing(10);

    btnApplyMatch = new QPushButton("Apply Matching", this);
    btnApplyMatch->setMinimumHeight(45);
    layoutMatch->addWidget(btnApplyMatch);

    layoutMatch->addStretch();
    paramsStackedWidget->addWidget(pageMatch);

    leftLayout->addWidget(paramsStackedWidget);
    leftLayout->addStretch();

    // ... الكود القديم بتاع زراير الـ Radio ...
    matchVBox->addWidget(radioSSD);
    // --- إضافة Sliders الـ Ratio ---
    ssdRatioSlider = new QSlider(Qt::Horizontal);
    ssdRatioSlider->setRange(1, 100);
    ssdRatioSlider->setValue(65); // القيمة الافتراضية 0.65
    ssdRatioValLbl = new QLabel("0.65");

    matchVBox->addWidget(new QLabel("SSD Ratio Test:"));
    QHBoxLayout *ssdLayout = new QHBoxLayout();
    ssdLayout->addWidget(ssdRatioSlider);
    ssdLayout->addWidget(ssdRatioValLbl);
    matchVBox->addLayout(ssdLayout);

    matchVBox->addSpacing(10);

    matchVBox->addWidget(radioNCC);

    nccRatioSlider = new QSlider(Qt::Horizontal);
    nccRatioSlider->setRange(1, 100);
    nccRatioSlider->setValue(65); // القيمة الافتراضية 0.65
    nccRatioValLbl = new QLabel("0.65");

    matchVBox->addWidget(new QLabel("NCC Ratio Test:"));
    QHBoxLayout *nccLayout = new QHBoxLayout();
    nccLayout->addWidget(nccRatioSlider);
    nccLayout->addWidget(nccRatioValLbl);
    matchVBox->addLayout(nccLayout);
    // ---------------------------------

    // --- Right Image Viewer Panel ---
    QVBoxLayout *rightLayout = new QVBoxLayout();
    QHBoxLayout *labelsLayout = new QHBoxLayout();
    inputLbl = new QLabel("Input", this);
    outputLbl = new QLabel("Output", this);
    inputLbl->setAlignment(Qt::AlignCenter); outputLbl->setAlignment(Qt::AlignCenter);
    inputLbl->setObjectName("boldLabel"); outputLbl->setObjectName("boldLabel");
    labelsLayout->addWidget(inputLbl); labelsLayout->addWidget(outputLbl);

    QHBoxLayout *imagesLayout = new QHBoxLayout();
    originalImageLabel = new QLabel(this); outputImageLabel = new QLabel(this);
    originalImageLabel->setObjectName("imageBox"); outputImageLabel->setObjectName("imageBox");
    originalImageLabel->setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Ignored); originalImageLabel->setScaledContents(true);
    outputImageLabel->setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Ignored); outputImageLabel->setScaledContents(true);
    imagesLayout->addWidget(originalImageLabel); imagesLayout->addWidget(outputImageLabel);

    timeLabel = new QLabel("Computational Time: 0.000 S", this);
    timeLabel->setObjectName("timeBox");
    timeLabel->setAlignment(Qt::AlignCenter);
    timeLabel->setFixedHeight(40);

    rightLayout->addLayout(labelsLayout, 0);
    rightLayout->addLayout(imagesLayout, 1);
    rightLayout->addWidget(timeLabel, 0);

    contentLayout->addLayout(leftLayout, 1);
    contentLayout->addLayout(rightLayout, 3);
    mainLayout->addLayout(contentLayout);

    // ==========================================
    // Event Connections
    // ==========================================
    connect(btnLoadMain, &QPushButton::clicked, this, &MainWindow::loadMainImage);
    connect(btnApplyHarris, &QPushButton::clicked, this, &MainWindow::applyHarris);
    connect(btnApplyLambda, &QPushButton::clicked, this, &MainWindow::applyLambda);

    connect(btnApplyMain, &QPushButton::clicked, this, &MainWindow::onApplySIFTOrMatch);
    // 4. تم إزالة السطر القديم btnApplyMatchLocal وربط أزرار الـ Matching الجديدة
    connect(btnLoadTemplate, &QPushButton::clicked, this, &MainWindow::loadTemplateImage);
    connect(btnApplyMatch, &QPushButton::clicked, this, &MainWindow::onApplySIFTOrMatch);

    connect(modeSelector, QOverload<int>::of(&QComboBox::currentIndexChanged), this, [this](int index){
        paramsStackedWidget->setCurrentIndex(index);

        // إذا تم اختيار Feature Matching (Mode index 2)
        if (index == 2) {
            // نغير الـ Labels لـ Image 1 و Image 2
            // ملاحظة: تأكدي من أسماء المتغيرات كما هي معرفة عندك (غالباً inputLbl و outputLbl)
            inputLbl->setText("Image 1");
            outputLbl->setText("Image 2");
        } else {
            // نرجع النصوص الأصلية لباقي الأوضاع (Harris/SIFT)
            inputLbl->setText("Input");
            outputLbl->setText("Output");
        }
    });
    // Harris Sliders
    connect(harrisThreshSlider, &QSlider::valueChanged, this, [this](int val){ harrisThreshValLbl->setText(QString::number(val / 100.0, 'f', 2)); });
    connect(harrisSigmaSlider, &QSlider::valueChanged, this, [this](int val){ harrisSigmaValLbl->setText(QString::number(val / 10.0, 'f', 1)); });
    connect(harrisWinSizeSlider, &QSlider::valueChanged, this, [this](int val){ harrisWinSizeValLbl->setText(QString::number((val * 2) + 1)); });
    connect(harrisKSlider, &QSlider::valueChanged, this, [this](int val){ harrisKValLbl->setText(QString::number(val / 100.0, 'f', 2)); });

    // Lambda Sliders
    connect(lambdaThreshSlider, &QSlider::valueChanged, this, [this](int val){ lambdaThreshValLbl->setText(QString::number(val / 100.0, 'f', 2)); });
    connect(lambdaSigmaSlider, &QSlider::valueChanged, this, [this](int val){ lambdaSigmaValLbl->setText(QString::number(val / 10.0, 'f', 1)); });
    connect(lambdaWinSizeSlider, &QSlider::valueChanged, this, [this](int val){ lambdaWinSizeValLbl->setText(QString::number((val * 2) + 1)); });

    // SIFT Sliders
    connect(siftOctavesSlider, &QSlider::valueChanged, this, [this](int val){ siftOctavesValLbl->setText(QString::number(val)); });
    connect(siftScalesSlider, &QSlider::valueChanged, this, [this](int val){ siftScalesValLbl->setText(QString::number(val)); });
    connect(siftSigmaSlider, &QSlider::valueChanged, this, [this](int val){ siftSigmaValLbl->setText(QString::number(val / 10.0, 'f', 1)); });
    connect(siftContrastThreshSlider, &QSlider::valueChanged, this, [this](int val){ siftContrastThreshValLbl->setText(QString::number(val / 100.0, 'f', 2)); });
    connect(siftEdgeThreshSlider, &QSlider::valueChanged, this, [this](int val){ siftEdgeThreshValLbl->setText(QString::number(val, 'f', 1)); });

    // Matching Sliders Connections
    connect(ssdRatioSlider, &QSlider::valueChanged, this, [this](int val){ ssdRatioValLbl->setText(QString::number(val / 100.0, 'f', 2)); });
    connect(nccRatioSlider, &QSlider::valueChanged, this, [this](int val){ nccRatioValLbl->setText(QString::number(val / 100.0, 'f', 2)); });
}

void MainWindow::applyStyle() {
    this->setStyleSheet(R"(
        QMainWindow, QWidget#centralWidget { background-color: #110918; color: #FFFFFF; font-family: "Segoe UI", Arial; }
        QLabel#appTitle { font-size: 26px; font-weight: bold; color: #D8B4FE; }
        QLabel#boldLabel { font-size: 16px; font-weight: bold; color: #E9D5FF; }
        QLabel { color: #FFFFFF; font-weight: bold; font-size: 13px; }
        QComboBox { background-color: #2D1B4E; color: white; border: 1px solid #581C87; border-radius: 5px; padding: 5px; font-weight: bold; }
        QComboBox QAbstractItemView { background-color: #2D1B4E; color: white; selection-background-color: #6B21A8; }
        QRadioButton { color: #E9D5FF; font-weight: bold; }
        QGroupBox { border: 1px solid #581C87; border-radius: 8px; margin-top: 15px; font-weight: bold; color: #D8B4FE; }
        QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top center; padding: 0 5px; }
        QPushButton { background-color: #6B21A8; color: white; border: none; border-radius: 8px; font-size: 14px; font-weight: bold; padding: 8px; }
        QPushButton:hover { background-color: #9333EA; }
        QPushButton:pressed { background-color: #A855F7; }
        QLabel#imageBox { background-color: #12081A; border: 2px solid #3B0764; border-radius: 8px; }
        QLabel#timeBox { background-color: #2D1B4E; color: #E9D5FF; font-size: 16px; font-weight: bold; border-radius: 8px; }
        QSlider::groove:horizontal { background: #2D1B4E; height: 6px; border-radius: 3px; }
        QSlider::sub-page:horizontal { background: #9333EA; border-radius: 3px; }
        QSlider::handle:horizontal { background: #D8B4FE; width: 14px; height: 14px; margin: -4px 0; border-radius: 7px; }
    )");
}

void MainWindow::displayImage(const cv::Mat& img, QLabel* label) {
    if(img.empty()) return;
    cv::Mat rgb;
    if (img.channels() == 3) cv::cvtColor(img, rgb, cv::COLOR_BGR2RGB);
    else cv::cvtColor(img, rgb, cv::COLOR_GRAY2RGB);
    QImage qimg((const unsigned char*)(rgb.data), rgb.cols, rgb.rows, rgb.step, QImage::Format_RGB888);
    label->setPixmap(QPixmap::fromImage(qimg));
}

void MainWindow::loadMainImage() {
    QString fileName = QFileDialog::getOpenFileName(this, "Open Image 1", "", "Images (*.png *.jpg *.jpeg *.bmp)");
    if (!fileName.isEmpty()) {
        currentImage = cv::imread(fileName.toStdString());
        if (!currentImage.empty()) {
            outputImageLabel->show(); // نرجع نظهر المربع اليمين

            displayImage(currentImage, originalImageLabel); // نعرض الصورة الأولى في الشمال

            // لو في صورة تانية متخزنة من قبل كدا، نعرضها في اليمين عشان منمسحهاش
            if (!templateImage.empty() && modeSelector->currentIndex() == 2) {
                displayImage(templateImage, outputImageLabel);
            } else {
                outputImageLabel->clear();
            }

            currentKeypoints.clear();
            timeLabel->setText("Image 1 Loaded.");
        }
    }
}

void MainWindow::applyHarris() {
    if (currentImage.empty()) { timeLabel->setText("⚠️ Please load an image first!"); return; }

    // --- التعديل هنا: تنظيف الشاشة وإرجاعها لحالتها الطبيعية ---
    outputImageLabel->show(); // إظهار المربع الأيمن
    displayImage(currentImage, originalImageLabel); // إرجاع الصورة الأصلية للمربع الأيسر
    // --------------------------------------------------------

    timeLabel->setText("⏳ Extracting Harris Features... Please Wait...");
    QApplication::setOverrideCursor(Qt::WaitCursor);
    QCoreApplication::processEvents();

    float threshold = harrisThreshSlider->value() / 100.0f;
    float sigma = harrisSigmaSlider->value() / 10.0f;
    int windowSize = (harrisWinSizeSlider->value() * 2) + 1;
    float k = harrisKSlider->value() / 100.0f;

    cv::TickMeter tm; tm.start();
    currentKeypoints = detector.detect(currentImage, CornerDetector::Method::HARRIS, threshold, windowSize, sigma, k);
    tm.stop();

    if (radioHarrisThreshold->isChecked()) {
        displayImage(detector.getThresholdMap(), outputImageLabel);
    } else {
        cv::Mat resultImg = detector.drawKeypoints(currentImage, currentKeypoints, false, cv::Scalar(0, 0, 255));
        displayImage(resultImg, outputImageLabel);
    }

    QApplication::restoreOverrideCursor();
    timeLabel->setText(QString("Computational Time: %1 S | Harris Keypoints: %2").arg(tm.getTimeSec(), 0, 'f', 3).arg(currentKeypoints.size()));
}

void MainWindow::applyLambda() {
    if (currentImage.empty()) { timeLabel->setText("⚠️ Please load an image first!"); return; }

    // --- التعديل هنا: تنظيف الشاشة وإرجاعها لحالتها الطبيعية ---
    outputImageLabel->show(); // إظهار المربع الأيمن
    displayImage(currentImage, originalImageLabel); // إرجاع الصورة الأصلية للمربع الأيسر
    // --------------------------------------------------------

    timeLabel->setText("⏳ Extracting Lambda Features... Please Wait...");
    QApplication::setOverrideCursor(Qt::WaitCursor);
    QCoreApplication::processEvents();

    float threshold = lambdaThreshSlider->value() / 100.0f;
    float sigma = lambdaSigmaSlider->value() / 10.0f;
    int windowSize = (lambdaWinSizeSlider->value() * 2) + 1;

    cv::TickMeter tm; tm.start();
    currentKeypoints = detector.detect(currentImage, CornerDetector::Method::SHI_TOMASI, threshold, windowSize, sigma, 0.04f);
    tm.stop();

    if (radioLambdaThreshold->isChecked()) {
        displayImage(detector.getThresholdMap(), outputImageLabel);
    } else {
        cv::Mat resultImg = detector.drawKeypoints(currentImage, currentKeypoints, false, cv::Scalar(0, 255, 0));
        displayImage(resultImg, outputImageLabel);
    }

    QApplication::restoreOverrideCursor();
    timeLabel->setText(QString("Computational Time: %1 S | Lambda Keypoints: %2").arg(tm.getTimeSec(), 0, 'f', 3).arg(currentKeypoints.size()));
}

void MainWindow::onApplySIFTOrMatch() {
    if (currentImage.empty()) { timeLabel->setText("⚠️ Please load an image first!"); return; }

    // --- التعديل هنا: تنظيف الشاشة وإرجاعها لحالتها الطبيعية ---
    outputImageLabel->show(); // إظهار المربع الأيمن
    displayImage(currentImage, originalImageLabel); // إرجاع الصورة الأصلية للمربع الأيسر
    // --------------------------------------------------------

    int mode = modeSelector->currentIndex();

    if (mode == 1) { // SIFT Mode

        timeLabel->setText("⏳ Extracting DoG Keypoints & Computing SIFT... Please Wait...");
        QApplication::setOverrideCursor(Qt::WaitCursor);
        QCoreApplication::processEvents();

        int octaves = siftOctavesSlider->value();
        int scales = siftScalesSlider->value();
        float initialSigma = siftSigmaSlider->value() / 10.0f;
        float contrastThresh = siftContrastThreshSlider->value() / 100.0f;
        float edgeThresh = siftEdgeThreshSlider->value();

        cv::TickMeter tm; tm.start();

        currentKeypoints = siftExtractor.detect(currentImage, octaves, scales, initialSigma, contrastThresh, edgeThresh);
        cv::Mat descriptors = siftExtractor.compute(currentImage, currentKeypoints, initialSigma, octaves);
        cv::Mat resultImg = siftExtractor.drawRichKeypoints(currentImage, currentKeypoints);

        tm.stop();

        displayImage(resultImg, outputImageLabel);
        QApplication::restoreOverrideCursor();

        timeLabel->setText(QString("Pure SIFT Time: %1 S | Keypoints: %2 | Descriptors: %3")
                               .arg(tm.getTimeSec(), 0, 'f', 3)
                               .arg(currentKeypoints.size())
                               .arg(descriptors.rows));
    }
    else if (mode == 2) { // Feature Matching Mode
        if (currentImage.empty() || templateImage.empty()) {
            timeLabel->setText("⚠️ Error: Load both Image 1 and Image 2!");
            return;
        }

        timeLabel->setText("⏳ Extracting Features & Matching... Please Wait");
        QApplication::setOverrideCursor(Qt::WaitCursor);
        QCoreApplication::processEvents();

        cv::TickMeter tm;
        tm.start();

        int octaves = siftOctavesSlider->value();
        int scales = siftScalesSlider->value();
        float initialSigma = siftSigmaSlider->value() / 10.0f;
        float contrastThresh = siftContrastThreshSlider->value() / 100.0f;
        float edgeThresh = siftEdgeThreshSlider->value();

        std::vector<cv::KeyPoint> kp1 = siftExtractor.detect(currentImage, octaves, scales, initialSigma, contrastThresh, edgeThresh);
        cv::Mat desc1 = siftExtractor.compute(currentImage, kp1, initialSigma, octaves);

        std::vector<cv::KeyPoint> kp2 = siftExtractor.detect(templateImage, octaves, scales, initialSigma, contrastThresh, edgeThresh);
        cv::Mat desc2 = siftExtractor.compute(templateImage, kp2, initialSigma, octaves);

        // قراءة قيم الـ Ratio من الـ Sliders وتحويلها لكسور (مثلا 65 هتبقى 0.65)
        float ssdRatio = ssdRatioSlider->value() / 100.0f;
        float nccRatio = nccRatioSlider->value() / 100.0f;

        // 3. إجراء المطابقة باستخدام SSD أو NCC
        std::vector<cv::DMatch> matches;
        if (!desc1.empty() && !desc2.empty()) {
            if (radioSSD->isChecked()) {
                matches = matchFeaturesSSD(desc1, desc2, ssdRatio);
            } else {
                matches = matchFeaturesNCC(desc1, desc2, nccRatio);
            }
        }
        tm.stop();

        if (!kp1.empty() && !kp2.empty() && !matches.empty()) {

            // --- التعديل السحري: توحيد أطوال الصور قبل الرسم ---
            cv::Mat drawImg1 = currentImage.clone();
            cv::Mat drawImg2 = templateImage.clone();
            std::vector<cv::KeyPoint> drawKp1 = kp1;
            std::vector<cv::KeyPoint> drawKp2 = kp2;

            // بنشوف مين الصورة الأطول فيهم عشان نخلي التانية طولها
            int targetHeight = std::max(drawImg1.rows, drawImg2.rows);

            // لو الصورة الأولى هي القصيرة، نكبرها ونظبط نقطها
            if (drawImg1.rows < targetHeight) {
                float scale = (float)targetHeight / drawImg1.rows;
                cv::resize(drawImg1, drawImg1, cv::Size(), scale, scale);
                for (auto& kp : drawKp1) { kp.pt.x *= scale; kp.pt.y *= scale; }
            }
            // لو الصورة التانية هي القصيرة، نكبرها ونظبط نقطها
            if (drawImg2.rows < targetHeight) {
                float scale = (float)targetHeight / drawImg2.rows;
                cv::resize(drawImg2, drawImg2, cv::Size(), scale, scale);
                for (auto& kp : drawKp2) { kp.pt.x *= scale; kp.pt.y *= scale; }
            }
            // ---------------------------------------------------

            cv::Mat imgMatches;
            // بنستخدم النسخ المتعدلة في الرسم بدل الصور الأصلية
            cv::drawMatches(drawImg1, drawKp1, drawImg2, drawKp2, matches, imgMatches,
                            cv::Scalar::all(-1), cv::Scalar::all(-1), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

            outputImageLabel->hide();
            displayImage(imgMatches, originalImageLabel);

            timeLabel->setText(QString("Success! Time: %1 S | Matches Found: %2").arg(tm.getTimeSec(), 0, 'f', 3).arg(matches.size()));
        } else {
            timeLabel->setText("❌ No Matches Found!");
        }

        QApplication::restoreOverrideCursor();
    }
}
