#pragma once
#include <QLabel>
#include <QImage>
#include <QPoint>
#include <QWheelEvent>
#include <QMouseEvent>

class ImageView : public QLabel {
    Q_OBJECT
public:
    explicit ImageView(QWidget *parent = nullptr);

    void setImage(const QImage &img); // задаёт изображение и сбрасывает трансформ
    bool hasImage() const { return !orig.isNull(); }

    // удобство: получить изображение, чтобы его отрисовать из OpenCV
    void setImageFromMat(const cv::Mat &mat);

protected:
    void wheelEvent(QWheelEvent *event) override;
    void mousePressEvent(QMouseEvent *event) override;
    void mouseMoveEvent(QMouseEvent *event) override;
    void mouseReleaseEvent(QMouseEvent *event) override;
    void resizeEvent(QResizeEvent *event) override;

private:
    QImage orig;
    double scale = 1.0;
    QPointF offset = {0,0}; // смещение в пикселях
    QPoint lastMouse;
    bool panning = false;

    void updatePixmap();
};
