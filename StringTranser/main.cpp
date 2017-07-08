#include "StringTranser.h"
#include <QtWidgets/QApplication>

int main(int argc, char *argv[])
{
	QApplication a(argc, argv);
	StringTranser w;
	w.show();
	return a.exec();
}
