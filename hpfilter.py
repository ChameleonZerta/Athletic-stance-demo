import scipy.signal as signal


def hpfilter():
    Fs = 50
    Fstop = 0.016
    Fpass = 0.032
    Astop = 60
    Apass = 1
    match = 'highpass'

    N, Wn = signal.cheb2ord(Fpass, Fstop, Apass, Astop, fs=Fs)
    sos = signal.cheby2(N, Astop, Wn, btype=match, output='sos')
    return sos
