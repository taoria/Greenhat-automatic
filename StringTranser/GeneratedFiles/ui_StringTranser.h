/********************************************************************************
** Form generated from reading UI file 'StringTranser.ui'
**
** Created by: Qt User Interface Compiler version 5.8.0
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_STRINGTRANSER_H
#define UI_STRINGTRANSER_H

#include <QtCore/QVariant>
#include <QtWidgets/QAction>
#include <QtWidgets/QApplication>
#include <QtWidgets/QButtonGroup>
#include <QtWidgets/QHeaderView>
#include <QtWidgets/QLabel>
#include <QtWidgets/QMainWindow>
#include <QtWidgets/QMenuBar>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QStatusBar>
#include <QtWidgets/QTextBrowser>
#include <QtWidgets/QToolBar>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_StringTranserClass
{
public:
    QWidget *centralWidget;
    QPushButton *pushButton;
    QLabel *label;
    QTextBrowser *textBrowser;
    QPushButton *greenhat;
    QMenuBar *menuBar;
    QToolBar *mainToolBar;
    QStatusBar *statusBar;

    void setupUi(QMainWindow *StringTranserClass)
    {
        if (StringTranserClass->objectName().isEmpty())
            StringTranserClass->setObjectName(QStringLiteral("StringTranserClass"));
        StringTranserClass->resize(978, 762);
        centralWidget = new QWidget(StringTranserClass);
        centralWidget->setObjectName(QStringLiteral("centralWidget"));
        pushButton = new QPushButton(centralWidget);
        pushButton->setObjectName(QStringLiteral("pushButton"));
        pushButton->setGeometry(QRect(10, 10, 75, 23));
        label = new QLabel(centralWidget);
        label->setObjectName(QStringLiteral("label"));
        label->setGeometry(QRect(30, 70, 181, 241));
        textBrowser = new QTextBrowser(centralWidget);
        textBrowser->setObjectName(QStringLiteral("textBrowser"));
        textBrowser->setGeometry(QRect(0, 30, 971, 661));
        greenhat = new QPushButton(centralWidget);
        greenhat->setObjectName(QStringLiteral("greenhat"));
        greenhat->setGeometry(QRect(90, 10, 71, 21));
        StringTranserClass->setCentralWidget(centralWidget);
        menuBar = new QMenuBar(StringTranserClass);
        menuBar->setObjectName(QStringLiteral("menuBar"));
        menuBar->setGeometry(QRect(0, 0, 978, 23));
        StringTranserClass->setMenuBar(menuBar);
        mainToolBar = new QToolBar(StringTranserClass);
        mainToolBar->setObjectName(QStringLiteral("mainToolBar"));
        StringTranserClass->addToolBar(Qt::TopToolBarArea, mainToolBar);
        statusBar = new QStatusBar(StringTranserClass);
        statusBar->setObjectName(QStringLiteral("statusBar"));
        StringTranserClass->setStatusBar(statusBar);

        retranslateUi(StringTranserClass);

        QMetaObject::connectSlotsByName(StringTranserClass);
    } // setupUi

    void retranslateUi(QMainWindow *StringTranserClass)
    {
        StringTranserClass->setWindowTitle(QApplication::translate("StringTranserClass", "StringTranser", Q_NULLPTR));
        pushButton->setText(QApplication::translate("StringTranserClass", "\351\200\211\346\213\251\346\226\207\344\273\266", Q_NULLPTR));
        label->setText(QString());
        greenhat->setText(QApplication::translate("StringTranserClass", "\347\273\277\345\270\275\346\234\272", Q_NULLPTR));
    } // retranslateUi

};

namespace Ui {
    class StringTranserClass: public Ui_StringTranserClass {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_STRINGTRANSER_H
