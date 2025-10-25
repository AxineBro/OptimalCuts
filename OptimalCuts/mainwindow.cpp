#include "mainwindow.h"
#include "ui_mainwindow.h"
#include <QDragEnterEvent>
#include <QMimeData>
#include <QFileInfo>
#include <QPixmap>
#include <QElapsedTimer>
#include <QJsonDocument>
#include <QJsonArray>
#include <QJsonObject>
#include <QFileDialog>
#include <QMessageBox>
#include <QWheelEvent>
#include <QResizeEvent>

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent), ui(new Ui::MainWindow) {
    ui->setupUi(this);
    setAcceptDrops(true);
    ui->labelOriginal->setScaledContents(false);
    ui->labelResult->setScaledContents(false);
}

MainWindow::~MainWindow() { delete ui; }

void MainWindow::dragEnterEvent(QDragEnterEvent *event) {
    if (event->mimeData()->hasUrls()) event->acceptProposedAction();
}

void MainWindow::dropEvent(QDropEvent *event) {
    QList<QUrl> urls = event->mimeData()->urls();
    if (urls.isEmpty()) return;
    QString path = urls.first().toLocalFile();
    if (path.isEmpty()) return;

    currentMat = cv::imread(path.toStdString(), cv::IMREAD_UNCHANGED);
    if (currentMat.empty()) return;

    originalImage = cvMatToQImage(currentMat);
    displayOriginal(originalImage);
    ui->labelResult->clear();
    ui->textInfo->clear();
    originalScale = 1.0;
    resultScale = 1.0;
}

void MainWindow::on_btnProcess_clicked() {
    if (currentMat.empty()) return;

    // Масштабируем изображение для скорости (если очень большое)
    cv::Mat workingMat;
    if (currentMat.cols > 2000 || currentMat.rows > 2000) {
        double scale = 0.5; // Масштабируем в 2 раза
        cv::resize(currentMat, workingMat, cv::Size(), scale, scale);
    } else {
        workingMat = currentMat.clone();
    }

    cv::Mat gray;
    if (workingMat.channels() > 1) cv::cvtColor(workingMat, gray, cv::COLOR_BGR2GRAY);
    else gray = workingMat.clone();

    cv::Mat bin;
    cv::threshold(gray, bin, 127, 255, cv::THRESH_BINARY);

    QElapsedTimer timer;
    timer.start();

    // Используем бОльшую аппроксимацию для скорости
    processor.process(bin, 3.0);

    qint64 ms = timer.elapsed();

    cv::Mat out;
    cv::cvtColor(bin, out, cv::COLOR_GRAY2BGR);

    // ТОЛЬКО внешние контуры и разрезы (минимальная визуализация)
    for (size_t i = 0; i < processor.contours.size(); ++i) {
        if (processor.hierarchy[i][3] == -1) { // Только внешние
            cv::polylines(out, processor.contours[i], true, cv::Scalar(0, 255, 0), 2);
        }
    }

    // Разрезы красным
    for (auto &cut : processor.cuts) {
        cv::line(out, cut.p_out, cut.p_in, cv::Scalar(0, 0, 255), 2);
    }

    displayResult(out);

    ui->textInfo->setText(QString::fromStdString(processor.getInfoString()) +
                          "\nProcessed in " + QString::number(ms) + " ms");

    ui->statusbar->showMessage(QString("Processed in %1 ms").arg(ms));
}

void MainWindow::on_btnExport_clicked() {
    if (processor.contours.empty()) return;

    QString fileName = QFileDialog::getSaveFileName(this, "Export JSON", "", "JSON (*.json)");
    if (fileName.isEmpty()) return;

    auto merged = processor.mergedContours();
    QJsonArray contoursArray;
    for (auto &vec : merged) {
        QJsonArray pointsArray;
        for (auto &p : vec) {
            QJsonObject pointObj;
            pointObj["x"] = p.x;
            pointObj["y"] = p.y;
            pointsArray.append(pointObj);
        }
        contoursArray.append(pointsArray);
    }

    QJsonObject root;
    root["mergedContours"] = contoursArray;

    // Исправленная часть: используем qint64 для sum
    qint64 totalPoints = 0;
    for (const auto& vec : merged) {
        totalPoints += static_cast<qint64>(vec.size());
    }
    root["totalPoints"] = totalPoints;

    QFile file(fileName);
    if (file.open(QIODevice::WriteOnly)) {
        file.write(QJsonDocument(root).toJson());
        file.close();
    }
}

void MainWindow::displayOriginal(const QImage &img) {
    QPixmap pix = QPixmap::fromImage(img).scaled(ui->labelOriginal->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation);
    ui->labelOriginal->setPixmap(pix);
}

void MainWindow::displayResult(const cv::Mat &mat) {
    QImage img = cvMatToQImage(mat);
    QPixmap pix = QPixmap::fromImage(img).scaled(ui->labelResult->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation);
    ui->labelResult->setPixmap(pix);
}

QImage MainWindow::cvMatToQImage(const cv::Mat &mat) {
    if (mat.type() == CV_8UC3) {
        cv::Mat rgb;
        cv::cvtColor(mat, rgb, cv::COLOR_BGR2RGB);
        return QImage((const unsigned char*)rgb.data, rgb.cols, rgb.rows, rgb.step, QImage::Format_RGB888).copy();
    } else if (mat.type() == CV_8UC1) {
        return QImage((const unsigned char*)mat.data, mat.cols, mat.rows, mat.step, QImage::Format_Indexed8).copy();
    } else if (mat.type() == CV_8UC4) {
        return QImage((const unsigned char*)mat.data, mat.cols, mat.rows, mat.step, QImage::Format_ARGB32).copy();
    }
    return QImage();
}

void MainWindow::wheelEvent(QWheelEvent *event) {
    if (event->angleDelta().y() > 0) {
        originalScale *= 1.1;
        resultScale *= 1.1;
    } else {
        originalScale /= 1.1;
        resultScale /= 1.1;
    }
    displayOriginal(originalImage);
    displayResult(cv::Mat());
}

void MainWindow::resizeEvent(QResizeEvent *event) {
    QMainWindow::resizeEvent(event);
    displayOriginal(originalImage);
    displayResult(cv::Mat());
}
