#pragma once
#include <QMainWindow>
#include <QImage>
#include <opencv2/opencv.hpp>
#include "imageprocessor.h"

QT_BEGIN_NAMESPACE
namespace Ui { class MainWindow; }
QT_END_NAMESPACE

class MainWindow : public QMainWindow {
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow() override;

protected:
    void dragEnterEvent(QDragEnterEvent *event) override;
    void dropEvent(QDropEvent *event) override;
    void wheelEvent(QWheelEvent *event) override;
    void resizeEvent(QResizeEvent *event) override;

private slots:
    void on_btnProcess_clicked();
    void on_btnExport_clicked();

private:
    Ui::MainWindow *ui;
    QImage originalImage;
    QImage resultImage;
    cv::Mat currentMat;
    ImageProcessor processor;

    double originalScale = 1.0;
    double resultScale = 1.0;
    double infoHeight = 200; // Начальная высота панели информации

    void displayOriginal(const QImage &img);
    void displayResult(const cv::Mat &mat);
    static QImage cvMatToQImage(const cv::Mat &mat);
    void updateScales();
};
