import matplotlib.pyplot as plt
from getRawAcceleration import *
from hpfilter import *
import scipy.signal as signal


def plotCorrActivityComparisonForSubject(subject, act1name, act2name):
    labels = ['Walking', 'WalkingUpstairs', 'WalkingDownstairs',
              'Sitting', 'Standing', 'Laying']
    actname = [act1name, act2name]
    acc, actid, t, fs = getRawAcceleration(SubjectID=subject, Component='x')

    sos = hpfilter()
    ab = signal.sosfilt(sos, acc)
    for j in range(1, 3):
        for k in range(1, 7):
            if actname[j - 1] == labels[k - 1]:
                ls = [x for x in [i for i, la in enumerate(actid) if la == k] if t[x] < 250]
                abw = [ab[x] for x in ls]
                corr = signal.correlate(abw, abw)
                lags = signal.correlation_lags(len(abw), len(abw))
                tc = (1 / fs) * lags
                plt.plot(tc, corr,
                         label=labels[k - 1])
                plt.xlim(-5, 5)
    plt.legend(loc='best')
    plt.title('Autocorrrelation Comparison')
    plt.grid()
    plt.show()
