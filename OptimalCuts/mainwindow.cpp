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
#include <iostream>
#include <set>

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent), ui(new Ui::MainWindow) {
    ui->setupUi(this);
    setAcceptDrops(true);
    ui->labelOriginal->setScaledContents(false);
    ui->labelResult->setScaledContents(false);
}

MainWindow::~MainWindow() {
    delete ui;
}

void MainWindow::dragEnterEvent(QDragEnterEvent *event) {
    if (event->mimeData()->hasUrls()) {
        event->acceptProposedAction();
    }
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

    // Масштабируем изображение для скорости на больших размерах
    cv::Mat workingMat;
    if (currentMat.cols > 2000 || currentMat.rows > 2000) {
        double scale = 0.5;
        cv::resize(currentMat, workingMat, cv::Size(), scale, scale);
    } else {
        workingMat = currentMat.clone();
    }

    cv::Mat gray;
    if (workingMat.channels() > 1) {
        cv::cvtColor(workingMat, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = workingMat.clone();
    }

    // Простая бинаризация с порогом 127 (можно настроить)
    cv::Mat bin;
    cv::threshold(gray, bin, 127, 255, cv::THRESH_BINARY); // Без INV, предполагая, что объекты белые на черном фоне

    QElapsedTimer timer;
    timer.start();

    processor.process(bin, 127.0, 0.0); // Отключаем аппроксимацию

    qint64 ms = timer.elapsed();

    // Визуализация: рисуем контуры с учетом полной иерархии
    cv::Mat out;
    cv::cvtColor(bin, out, cv::COLOR_GRAY2BGR);

    // Массив цветов для 10 уровней
    std::vector<cv::Scalar> colors = {
        cv::Scalar(0, 255, 0),    // Уровень 1: Зеленый (внешний)
        cv::Scalar(255, 0, 0),    // Уровень 2: Красный
        cv::Scalar(0, 0, 255),    // Уровень 3: Синий
        cv::Scalar(255, 255, 0),  // Уровень 4: Желтый
        cv::Scalar(0, 255, 255),  // Уровень 5: Голубой
        cv::Scalar(255, 0, 255),  // Уровень 6: Фиолетовый
        cv::Scalar(128, 0, 0),    // Уровень 7: Темно-красный
        cv::Scalar(0, 128, 0),    // Уровень 8: Темно-зеленый
        cv::Scalar(0, 0, 128),    // Уровень 9: Темно-синий
        cv::Scalar(128, 128, 128) // Уровень 10: Серый
    };

    // Цвета в зависимости от уровня иерархии
    for (size_t i = 0; i < processor.contours.size(); ++i) {
        int level = 0;
        int current = (int)i;
        while (current != -1) {
            level++;
            current = processor.hierarchy[current][3]; // Идем вверх по родителям
        }
        cv::Scalar color = (level <= 10 && level > 0) ? colors[level - 1] : cv::Scalar(0, 0, 0); // Черный для >10
        if (!processor.contours[i].empty()) {
            cv::polylines(out, processor.contours[i], true, color, 2);
        }
    }

    displayResult(out);

    QString infoStr = QString::fromStdString(processor.getInfoString()) +
                      "\nОбработано за " + QString::number(ms) + " мс";
    ui->textInfo->setText(infoStr);

    ui->statusbar->showMessage(QString("Обработано за %1 мс").arg(ms));

    // Расширенный отладочный вывод
    std::cout << "=== Отладка иерархии (RETR_TREE) ===" << std::endl;
    for (size_t i = 0; i < processor.hierarchy.size(); ++i) {
        const auto& h = processor.hierarchy[i];
        std::cout << "Контур #" << i << ": ";
        std::cout << "Следующий: " << h[0] << ", Предыдущий: " << h[1] << ", Первый ребенок: " << h[2] << ", Родитель: " << h[3];
        int level = 0;
        int current = (int)i;
        while (current != -1) {
            level++;
            current = processor.hierarchy[current][3];
        }
        std::cout << ", Уровень: " << level << ", Точек: " << processor.contours[i].size() << std::endl;

        if (h[2] != -1) {
            std::cout << "  Дети: ";
            int child = h[2];
            while (child != -1) {
                std::cout << "#" << child << " ";
                child = processor.hierarchy[child][0];
            }
            std::cout << std::endl;
        }
    }
    std::cout << "=======================" << std::endl;
}

void MainWindow::on_btnExport_clicked() {
    if (processor.contours.empty()) return;

    QString fileName = QFileDialog::getSaveFileName(this, "Экспорт JSON", "", "JSON (*.json)");
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
