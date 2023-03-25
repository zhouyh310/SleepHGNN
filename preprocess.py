import os
import mne
import numpy as np
import scipy.io as sio
from sklearn import metrics


path = './data'


def existOrMakeDir(path):
    if not os.path.exists(path):
        os.mkdir(path)

existOrMakeDir(os.path.join(path, 'data'))
existOrMakeDir(os.path.join(path, 'label'))
existOrMakeDir(os.path.join(path, 'psd'))
existOrMakeDir(os.path.join(path, 'nmi_adj_mat'))


for sub_id in range(1, 11):
    dfile = os.path.join(path, 'raw_data', 'subject{}.mat'.format(sub_id))

    ext_data = sio.loadmat(dfile)
    keys = list(ext_data.keys())[3:]
    ext = np.zeros([len(keys), ext_data[keys[0]].shape[0], 6000])
    for i in keys:
        ext[keys.index(i)] = ext_data[i]
    ext = ext.swapaxes(0, 1)
    np.save(os.path.join(path, 'data', 'subject{}.npy'.format(sub_id)), ext)

    for expert_id in range(1, 3):
        devt = os.path.join(path, 'raw_label', '{}_{}.txt'.format(sub_id, expert_id))
        lines = np.loadtxt(devt, dtype=np.float64).astype(int)[:-30]
        lines = np.array([i if i < 4 else i - 1 for i in lines])
        np.save(os.path.join(path, 'label',
                '{}_{}.npy'.format(sub_id, expert_id)), lines)


ch_nameslist = ['F3_A2', 'C3_A2', 'O1_A2', 'F4_A1',
                'C4_A1', 'O2_A1', 'ROC_A1', 'LOC_A2', 'X1', 'X2', 'X3']

event_id = {'Sleep stage W': 0,
            'Sleep stage N1': 1,
            'Sleep stage N2': 2,
            'Sleep stage N3/4': 3,
            'Sleep stage R': 4}
info = mne.create_info(ch_names=ch_nameslist, sfreq=200, ch_types='eeg')


for sub_id in range(1, 11):
    print(f'Extractng PSD for subject {sub_id}...')
    pData = os.path.join(path, 'data', 'subject{}.npy'.format(sub_id))
    pLabel = os.path.join(path, 'label', '{}_1.npy'.format(sub_id))
    data = np.load(pData)
    label = np.load(pLabel)

    events = [[i * 30 * 200, 0, int(j)]
              for i, j in zip(range(len(label)), label)]
    epochs = mne.EpochsArray(data, info=info, events=events, event_id=event_id)

    spectrum = epochs.compute_psd(method="welch", fmin=0, fmax=100)
    psd = spectrum.get_data()

    np.save(os.path.join(path, 'psd', 'subject{}.npy'.format(sub_id)), psd)


def MI(x, bins=10):
    a = np.zeros(shape=(len(x), len(x)))
    for i in range(len(x)):
        for j in range(i, len(x)):
            y = np.histogram2d(x[i], x[j], bins=bins)[0]
            a[i][j] = metrics.mutual_info_score(None, None, contingency=y)
    b = a/a.max(axis=0)
    return a, b


threshold = 0.1
existOrMakeDir(os.path.join(path, 'nmi_adj_mat', f'threshold_{threshold}'))

for sub_id in range(1, 11):
    pData = os.path.join(path, 'data', 'subject{}.npy'.format(sub_id))
    data = np.load(pData)
    x = (data - np.mean(data)) / np.std(data)
    epoch, channel, length = x.shape
    x = np.swapaxes(x, 0, 1)
    x = x.reshape([channel, epoch * length])[:6, :]

    _, nni = MI(x, bins=10)
    up_tri_mat = np.triu(nni, 1)
    adj_mat = np.ones([6, 6]) * (up_tri_mat > threshold)

    np.save(os.path.join(path, 'nmi_adj_mat',
            f'threshold_{threshold}', f'subject_{sub_id}_adj_mat.npy'), adj_mat)
