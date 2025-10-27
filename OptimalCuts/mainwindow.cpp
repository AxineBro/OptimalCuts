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
#include <QMouseEvent>
#include <QPainter>
#include <iostream>
#include <set>

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent), ui(new Ui::MainWindow) {
    ui->setupUi(this);
    setAcceptDrops(true);
    ui->labelOriginal->setScaledContents(false);
    ui->labelResult->setScaledContents(false);

    // Устанавливаем курсоры для индикации возможности панорамирования
    ui->labelOriginal->setCursor(Qt::OpenHandCursor);
    ui->labelResult->setCursor(Qt::OpenHandCursor);
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
    resetView();
    displayImages();
    ui->textInfo->clear();
}

void MainWindow::on_btnProcess_clicked() {
    if (currentMat.empty()) return;

    cv::Mat workingMat = currentMat.clone();

    cv::Mat gray;
    if (workingMat.channels() > 1) {
        cv::cvtColor(workingMat, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = workingMat.clone();
    }

    // Простая бинаризация с порогом 127
    cv::Mat bin;
    cv::threshold(gray, bin, 127, 255, cv::THRESH_BINARY);

    QElapsedTimer timer;
    timer.start();

    processor.process(bin, 127.0, 0.0);

    qint64 ms = timer.elapsed();

    // Визуализация: рисуем контуры и разрезы
    cv::Mat out;
    cv::cvtColor(bin, out, cv::COLOR_GRAY2BGR);

    // Рисуем все контуры зеленым цветом
    cv::Scalar greenColor(0, 255, 0);
    for (size_t i = 0; i < processor.contours.size(); ++i) {
        if (!processor.contours[i].empty()) {
            cv::polylines(out, processor.contours[i], true, greenColor, 2);
        }
    }

    // Рисуем разрезы красным цветом
    cv::Scalar redColor(0, 0, 255);
    for (const auto& cut : processor.cuts) {
        cv::line(out, cut.p_out, cut.p_in, redColor, 2);
        // Добавляем точки для лучшей видимости
        cv::circle(out, cut.p_out, 3, redColor, -1);
        cv::circle(out, cut.p_in, 3, redColor, -1);
    }

    resultImage = cvMatToQImage(out);
    displayImages();

    QString infoStr = QString::fromStdString(processor.getInfoString()) +
                      "\nОбработано за " + QString::number(ms) + " мс";
    ui->textInfo->setText(infoStr);

    ui->statusbar->showMessage(QString("Обработано за %1 мс").arg(ms));
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
        QMessageBox::information(this, "Успех", "Файл успешно экспортирован");
    } else {
        QMessageBox::warning(this, "Ошибка", "Не удалось сохранить файл");
    }
}

void MainWindow::on_btnResetView_clicked() {
    resetView();
    displayImages();
}

void MainWindow::displayImages() {
    if (!originalImage.isNull()) {
        QPixmap pix = getScaledPixmap(originalImage, ui->labelOriginal);
        ui->labelOriginal->setPixmap(pix);
    }

    if (!resultImage.isNull()) {
        QPixmap pix = getScaledPixmap(resultImage, ui->labelResult);
        ui->labelResult->setPixmap(pix);
    }
}

QPixmap MainWindow::getScaledPixmap(const QImage &img, QLabel *label) {
    if (img.isNull() || !label) return QPixmap();

    // Создаем изображение с учетом масштаба и смещения
    int scaledWidth = int(img.width() * scale);
    int scaledHeight = int(img.height() * scale);

    // Масштабируем изображение
    QImage scaledImg = img.scaled(scaledWidth, scaledHeight, Qt::KeepAspectRatio, Qt::SmoothTransformation);

    // Создаем pixmap размера label
    QPixmap pixmap(label->size());
    pixmap.fill(Qt::gray);

    QPainter painter(&pixmap);
    painter.setRenderHint(QPainter::SmoothPixmapTransform);

    // Вычисляем позицию для отрисовки с учетом смещения
    int x = offset.x();
    int y = offset.y();

    // Ограничиваем смещение, чтобы не выходить за границы
    QRect labelRect = label->rect();

    // Центрируем если изображение меньше label
    if (scaledWidth < labelRect.width()) {
        x = (labelRect.width() - scaledWidth) / 2;
    } else {
        // Ограничиваем смещение для больших изображений
        x = qMax(labelRect.width() - scaledWidth, qMin(x, 0));
    }

    if (scaledHeight < labelRect.height()) {
        y = (labelRect.height() - scaledHeight) / 2;
    } else {
        // Ограничиваем смещение для больших изображений
        y = qMax(labelRect.height() - scaledHeight, qMin(y, 0));
    }

    // Рисуем масштабированное изображение
    painter.drawImage(x, y, scaledImg);

    // Добавляем информацию о масштабе
    painter.save();
    painter.setPen(Qt::white);
    painter.setBrush(QColor(0, 0, 0, 128));
    QFont font = painter.font();
    font.setPointSize(8);
    painter.setFont(font);

    QString info = QString("Масштаб: %1%").arg(int(scale * 100));
    QRect textRect = painter.fontMetrics().boundingRect(info);
    textRect.adjust(-4, -2, 4, 2);
    textRect.moveTo(5, 5);

    painter.drawRect(textRect);
    painter.drawText(textRect, Qt::AlignCenter, info);
    painter.restore();

    return pixmap;
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
    if (originalImage.isNull()) return;

    QPoint numDegrees = event->angleDelta() / 8;

    if (!numDegrees.isNull()) {
        double zoomFactor = 1.1;

        // Определяем позицию курсора относительно активного label
        QPoint mousePos;
        if (ui->labelOriginal->underMouse()) {
            mousePos = ui->labelOriginal->mapFromParent(event->position().toPoint());
        } else if (ui->labelResult->underMouse()) {
            mousePos = ui->labelResult->mapFromParent(event->position().toPoint());
        } else {
            return;
        }

        // Вычисляем позицию в координатах изображения до масштабирования
        QPointF imagePosBefore((mousePos.x() - offset.x()) / scale,
                               (mousePos.y() - offset.y()) / scale);

        // Применяем масштабирование
        if (numDegrees.y() > 0) {
            scale *= zoomFactor;
        } else {
            scale /= zoomFactor;
        }

        // Ограничиваем масштаб
        scale = qMax(0.1, qMin(scale, 10.0));

        // Корректируем смещение для zoom к курсору
        offset.setX(mousePos.x() - imagePosBefore.x() * scale);
        offset.setY(mousePos.y() - imagePosBefore.y() * scale);

        displayImages();
    }

    event->accept();
}

void MainWindow::mousePressEvent(QMouseEvent *event) {
    if (event->button() == Qt::LeftButton) {
        if ((ui->labelOriginal->underMouse() || ui->labelResult->underMouse()) &&
            !originalImage.isNull()) {
            isDragging = true;
            lastDragPos = event->pos();
            ui->labelOriginal->setCursor(Qt::ClosedHandCursor);
            ui->labelResult->setCursor(Qt::ClosedHandCursor);
        }
    }
    QMainWindow::mousePressEvent(event);
}

void MainWindow::mouseMoveEvent(QMouseEvent *event) {
    if (isDragging && !originalImage.isNull()) {
        QPoint delta = event->pos() - lastDragPos;
        offset += delta;
        lastDragPos = event->pos();
        displayImages();
    }
    QMainWindow::mouseMoveEvent(event);
}

void MainWindow::mouseReleaseEvent(QMouseEvent *event) {
    if (event->button() == Qt::LeftButton && isDragging) {
        isDragging = false;
        ui->labelOriginal->setCursor(Qt::OpenHandCursor);
        ui->labelResult->setCursor(Qt::OpenHandCursor);
    }
    QMainWindow::mouseReleaseEvent(event);
}

void MainWindow::resizeEvent(QResizeEvent *event) {
    QMainWindow::resizeEvent(event);
    displayImages();
}

void MainWindow::resetView() {
    scale = 1.0;
    offset = QPoint(0, 0);
    ui->labelOriginal->setCursor(Qt::OpenHandCursor);
    ui->labelResult->setCursor(Qt::OpenHandCursor);
}
