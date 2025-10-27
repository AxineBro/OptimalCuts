QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

CONFIG += c++17

SOURCES += \
    imageprocessor.cpp \
    main.cpp \
    mainwindow.cpp

HEADERS += \
    imageprocessor.h \
    mainwindow.h

FORMS += \
    mainwindow.ui

qnx: target.path = /tmp/$${TARGET}/bin
else: unix:!android: target.path = /opt/$${TARGET}/bin
!isEmpty(target.path): INSTALLS += target

# === OpenCV ===
INCLUDEPATH += Z:/Library/opencv/build/include

CONFIG(release, debug|release) {
    LIBS += -LZ:/Library/opencv/build/x64/vc16/lib \
            -lopencv_world4120
}

CONFIG(debug, debug|release) {
    LIBS += -LZ:/Library/opencv/build/x64/vc16/lib \
            -lopencv_world4120d
}
