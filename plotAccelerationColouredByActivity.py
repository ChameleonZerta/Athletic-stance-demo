import matplotlib.pyplot as plt


def plotAccelerationColouredByActivity(t, acc, actid, title):
    acc_1, acc_2, acc_3, acc_4, acc_5, acc_6 = [[] for _ in range(6)]
    d = {1: acc_1, 2: acc_2, 3: acc_3, 4: acc_4, 5: acc_5, 6: acc_6}
    labels = ['Walking', 'WalkingUpstairs', 'WalkingDownstairs',
              'Sitting', 'Standing', 'Laying']
    for k in range(1, 7):
        if title == labels[k - 1]:
            plt.plot(t, acc,
                     label=labels[k - 1])
            plt.xlim(t[0], t[-1])
            break
        for (i, j) in list(zip(actid, acc)):
            dc = d.pop(k)
            if k == i:
                dc.append(j)
            else:
                dc.append(None)
            d.update({k: dc})
        plt.plot(t, d[k],
                 label=labels[k - 1])
        plt.xlim(0, t[-1])
    plt.legend(loc='best')
    plt.title(title)
    plt.grid()


def plotAccelerationColouredByActivity2(t, acc, ab, actid, title1, title2):
    fig = plt.figure()
    plt.subplot(2, 1, 1)
    plotAccelerationColouredByActivity(t, acc, actid, title1)
    plt.subplot(2, 1, 2)
    plotAccelerationColouredByActivity(t, ab, actid, title2)
    fig.tight_layout(pad=0.4, w_pad=0, h_pad=0)
    plt.show()
