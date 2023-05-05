import matplotlib.pyplot as plt
from getRawAcceleration import *
from hpfilter import *
import scipy.signal as signal
import numpy as np


def plotPSDForGivenActivity(actname):
    labeln = 0
    labels = ['Walking', 'WalkingUpstairs', 'WalkingDownstairs',
              'Sitting', 'Standing', 'Laying']
    for k in range(1, 7):
        if actname == labels[k - 1]:
            labeln = k
            break
    ax = plt.axes(projection='3d')
    for ii in range(1, 31):
        subject = 31 - ii
        acc, actid, t, fs = getRawAcceleration(SubjectID=subject, Component='x')
        sos = hpfilter()
        ab = signal.sosfilt(sos, acc)

        ls = [x for x in [i for i, la in enumerate(actid) if la == labeln] if t[x] < 250]
        abw = [ab[x] for x in ls]

        xli = subject
        f, Pxx = signal.welch(abw, fs, nperseg=512, nfft=2048)
        xli = xli * np.ones(len(f))
        ax.plot(xli, f, Pxx)
    ax.set_xlim(31, 0)
    plt.xlabel('SubjectID')
    plt.ylabel('frequency (Hz)')
    plt.show()
