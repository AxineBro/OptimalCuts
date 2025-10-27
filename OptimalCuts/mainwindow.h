#pragma once
#include <QMainWindow>
#include <QImage>
#include <QPoint>
#include <QWheelEvent>
#include <QMouseEvent>
#include <QLabel>
#include <QPainter>
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
    void mousePressEvent(QMouseEvent *event) override;
    void mouseMoveEvent(QMouseEvent *event) override;
    void mouseReleaseEvent(QMouseEvent *event) override;

private slots:
    void on_btnProcess_clicked();
    void on_btnExport_clicked();
    void on_btnResetView_clicked();

private:
    Ui::MainWindow *ui;
    QImage originalImage;
    QImage resultImage;
    cv::Mat currentMat;
    ImageProcessor processor;

    double scale = 1.0;
    QPoint offset;
    QPoint lastDragPos;
    bool isDragging = false;

    void displayImages();
    static QImage cvMatToQImage(const cv::Mat &mat);
    void resetView();
    QPixmap getScaledPixmap(const QImage &img, QLabel *label);
};
