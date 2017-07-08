#pragma once

#include <QtWidgets/QMainWindow>
#include "ui_StringTranser.h"
#include <QFileDialog>
class StringTranser : public QMainWindow
{
	Q_OBJECT

public:
	StringTranser(QWidget *parent = Q_NULLPTR);

private:
	Ui::StringTranserClass ui;
	QString  image_path;
	QPixmap *qpixmap;
public slots:
	void OpenImageFile();
	void OpenGreenFile();
	void RefreshTextBrowser(QString qsrt);
};
