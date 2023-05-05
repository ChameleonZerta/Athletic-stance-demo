import matplotlib.pyplot as plt


def plotCompareHistForActivities(acc, actid, label1, label2):
    def plotHistogram(acc, actid, label):
        colorn = 0
        actids = [1, 2, 3, 4, 5, 6]
        labels = ['Walking', 'WalkingUpstairs', 'WalkingDownstairs',
                  'Sitting', 'Standing', 'Laying']
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        acc_l = []
        for (actidn, labeln, color) in list(zip(actids, labels, colors)):
            if label == labeln:
                colorn = color
                for (i, j) in list(zip(actid, acc)):
                    if i == actidn:
                        acc_l.append(j)
        n, bins, patches = plt.hist(acc_l, 40, (0, 20), color=colorn, edgecolor='k', label=label)
        plt.legend(loc="best")
        return n, bins, patches

    plt.figure()
    plt.subplot(2, 1, 1)
    plotHistogram(acc, actid, label1)
    plt.grid()

    plt.subplot(2, 1, 2)
    plotHistogram(acc, actid, label2)
    plt.grid()
    plt.show()
