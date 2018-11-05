#include <QCoreApplication>
#include "loadterminal.hpp"

int main(int argc, char *argv[])
{
    QCoreApplication a(argc, argv);

    LoadTerminal *ter = new LoadTerminal(&a);

    QObject::connect(ter, SIGNAL(finished()), &a, SLOT(quit()));

    QTimer::singleShot(0, ter, SLOT(run()));

    return a.exec();
}
