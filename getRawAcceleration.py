import scipy.io as scio
import numpy as np


def getRawAcceleration(*, SubjectID, Component):
    data = scio.loadmat('./Data/RecordedAccelerationsBySubject.mat')
    # data = scio.loadmat('./Data/BufferedAccelerations.mat')

    subjects = data['subjects']
    actid = subjects[0, SubjectID - 1][0][:, 0]

    if Component == 'x':
        component_n = 0
    elif Component == 'y':
        component_n = 1
    else:
        component_n = 2

    acc = subjects[0, SubjectID - 1][1][:, component_n]
    fs = data['fs'][0][0]
    t = (1 / data['fs'] * np.arange(0, len(acc))).T[:, 0]

    return acc, actid, t, fs
