import matplotlib.pyplot as plt
from getRawAcceleration import *
from hpfilter import *
import scipy.signal as signal


def plotPSDActivityComparisonForSubject(subject, act1name, act2name):
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
                f, Pxx = signal.welch(abw, fs, nperseg=512, nfft=2048)
                plt.semilogy(f, Pxx)
                plt.xlim(0, 10)
                plt.ylim(10 ** -5, 10 ** 2)
    plt.title('Power Spectral Density Comparison')
    plt.grid()
    plt.show()
