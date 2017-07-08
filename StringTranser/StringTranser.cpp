#include "StringTranser.h"
#include<QtDebug>
#include <QPainter>
#include <QLabel>
#include<stdio.h>
#include <fstream>
#include<QTextBrowser>
#include<QRgb>
#include <opencv2/core/core.hpp>  
#include <opencv2/highgui/highgui.hpp>  
#include <opencv2/imgproc/imgproc.hpp>  
#include <opencv2/objdetect/objdetect.hpp>
#include <iostream>
#include<sstream>
#include<QRunnable>
#include<QThreadPool>
#include <filesystem>
using namespace cv;
const std::string green_hat_address = "D:\\greenhat.jpg"; //greenhat 's location
char buff[13] = { '@','$','#' ,'&','W','O','w','v','o' ,';',',','.' ,'.' };
class GreenRun :public QRunnable {
	Mat& greenPNG(std::string path);
	QString path;

public:
	Mat hat = imread(green_hat_address);
	std::stringstream ss;
	GreenRun(const QString& path);

	~GreenRun();
	void run() override;
};
StringTranser::StringTranser(QWidget *parent)
	: QMainWindow(parent)
{
	ui.setupUi(this);
	connect(ui.pushButton, SIGNAL(clicked()), this,SLOT(OpenImageFile()) );
	connect(ui.greenhat, SIGNAL(clicked()), this, SLOT(OpenGreenFile()));
}
class Runnable :public QRunnable

{
protected:
	char RGBToChar(QRgb qrgb);
	QTextBrowser *qtb;
	std::string str;
	void colorReduce9(cv::Mat& image, int nums, int div = 64);

	Mat getFrame(std::string avipath, int maxNum);

	
	QString path;
public:
	std::stringstream ss;
	Runnable(QString path,QTextBrowser *qtb);

	~Runnable();



	void run() override;

protected:
	
private:

};


char Runnable::RGBToChar(QRgb qrgb) {
	//1/5 2/5 1/4
	unsigned int temp = qrgb & 0x00FFFFFF;
	int red = ((temp & 0xff0000) >> 16) / 10 * 3;
	int green = ((temp & 0x00ff00) >> 8) / 10 * 3;
	int blue = ((temp & 0x0000ff)) / 10 * 4;
	int ans = red + green + blue;
	int level = 255 / 12;
	return buff[ans / level];
}

void Runnable::colorReduce9(cv::Mat& image, int nums, int div) {
	// div must be a power of 2  
	
	int n = static_cast<int>(log(static_cast<double>(div)) / log(2.0));
	// mask used to round the pixel value  
	uchar mask = 0xFF << n; // e.g. for div=16, mask= 0xF0  
	// get iterators  
	ss.flush();
	ss.str("");
	int count = 0;
	Size dsize = Size(image.cols*0.3, image.rows*0.2);
	Mat image2 = Mat(Size(150,50), CV_32F);
	resize(image,image2,Size(150,50));
	cv::Mat_<cv::Vec3b>::iterator it = image2.begin<cv::Vec3b>();
	cv::Mat_<cv::Vec3b>::iterator itend = image2.end<cv::Vec3b>();
	for (; it != itend; ++it) {
		(*it)[0] = (*it)[0] & mask + div / 2;
		(*it)[1] = (*it)[1] & mask + div / 2;
		(*it)[2] = (*it)[2] & mask + div / 2;
		ss<<RGBToChar((*it)[0] | (static_cast<unsigned int>((*it)[1])) << 8 | (static_cast<unsigned int>((*it)[2]) << 16));
		if (++count % image2.cols == 0) {
			ss << "\n";
		}
		str = ss.str();
		
	}
}

Mat Runnable::getFrame(std::string avipath, int maxNum) {
	Mat frame;
	cv::VideoCapture capture(avipath);
	long totalFrameNumber = capture.get(CV_CAP_PROP_FRAME_COUNT);
	qtb->setText("");
	if (!capture.grab()) {

		std::cout << "grab in opencv failed!" << endl;
		return frame;
	}
	int VideoPos = 0;
	while (maxNum >= VideoPos) {
		capture >> frame;
		VideoPos++;
		if (frame.empty()) {
			return frame;
		}
		else {
			colorReduce9(frame, VideoPos);
		}
		_sleep(5);
		qtb->setText(QString::fromStdString(str));
		qtb->repaint();
		
	}
	return frame;
}

Runnable::Runnable(QString path,QTextBrowser *qtb) {
	this->path = path;
	this->qtb = qtb;
}

Runnable::~Runnable() {
}

void Runnable::run() {
	getFrame(path.toStdString(),2000);
}
void greening(cv::Mat &image,Point startPoint,Mat &greenHat) {

	int nr = image.rows; // number of rows  
	int nc = image.cols;
	for (int j = startPoint.y; j<startPoint.y+greenHat.rows; j++) {
		if (j < 0 || j>image.rows) continue;
		for (int i = startPoint.x; i<startPoint.x+greenHat.cols; i++) {
			if (i<0 || i>image.cols) continue;
			if(!(greenHat.at<Vec3b>(j-startPoint.y,i-startPoint.x)[1] >233 && greenHat.at<Vec3b>(j - startPoint.y, i - startPoint.x)[0]>233 && greenHat.at<Vec3b>(j - startPoint.y, i - startPoint.x)[2]>233)) {
				image.at<Vec3b>(j, i) = greenHat.at<Vec3b>(j - startPoint.y, i - startPoint.x);
			}
		}
	}
}
Mat& GreenRun::greenPNG(std::string tpath) {
	CascadeClassifier face_cascade;
	Mat frame = imread(tpath);
	
	Mat frame_gray;
	std::string face_cascade_name = "D:\\OpenCV\\opencv\\build\\etc\\haarcascades\\haarcascade_frontalface_alt.xml";
	qDebug() << QString::fromStdString(tpath)<< endl;
	std::vector<Rect> faces;
	
	if (!face_cascade.load(face_cascade_name)) { qDebug() << "Error Loading files" << endl;; return frame; };
	qDebug() << faces.size() << endl;
	cvtColor(frame, frame_gray, CV_BGR2GRAY);
	equalizeHist(frame_gray, frame_gray);
	face_cascade.detectMultiScale(frame_gray, faces, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(30, 30));
	qDebug() << faces.size() << endl;

	for (size_t i = 0; i < faces.size(); i++){
		Size dsize = Size(faces[i].width*1.5, faces[i].height*1.1);
		Mat resizedHat = Mat(dsize, CV_32S);
		resize(hat, resizedHat, dsize);
		Point startPoint(faces[i].x - faces[i].width*0.25, faces[i].y-faces[i].height*0.6);
		greening(frame, startPoint, resizedHat);
	}
	imwrite(tpath+"233.jpg", frame);
	return frame;
}


GreenRun::GreenRun(const QString& path) {
	this->path = path;
}

GreenRun::~GreenRun() {
}

void GreenRun::run() {
	this->greenPNG(path.toStdString());
}
void StringTranser::OpenImageFile() {
	QString path = QFileDialog::getOpenFileName(this, tr("Open Video"), ".", tr("Image Files(*.mp4)"));
	Runnable* rrr = new Runnable(path,ui.textBrowser);
	if(path=="") {
		return; 
	}
/*	std::ofstream myfile;
	char filename[100];
	sprintf(filename, "file.html");
	myfile.open(filename);
	myfile << "<meta http-equiv=\"Content - Type\" content=\"text / html; charset = gb2312\" /> ";
	myfile << "<style> \n body \n {\n font-family:monospace;font-szie:8px \n} \n </style> ";
	myfile << "<body>";
	*/
	rrr->run();
}
void StringTranser::OpenGreenFile() {
	QString path = QFileDialog::getOpenFileName(this, tr("Open Image"), ".", tr("Image Files(*.png *.jpg)"));
	GreenRun* rrr = new GreenRun(path);
	if(path=="") {
		return; 
	}
	/*	std::ofstream myfile;
	char filename[100];
	sprintf(filename, "file.html");
	myfile.open(filename);
	myfile << "<meta http-equiv=\"Content - Type\" content=\"text / html; charset = gb2312\" /> ";
	myfile << "<style> \n body \n {\n font-family:monospace;font-szie:8px \n} \n </style> ";
	myfile << "<body>";
	*/
	rrr->run();
}
void StringTranser::RefreshTextBrowser(QString qstr) {
	ui.textBrowser->setText(qstr);
	ui.textBrowser->repaint();
	
}


