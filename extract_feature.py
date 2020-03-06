import numpy as np
from scipy.io import wavfile
from scipy import signal
import matplotlib.pyplot as plt


def pre_emphesis(data, alpha=0.97):
    return np.append(data[0], data[1:] - alpha*data[:-1])


def log_spectrogram(data, sample_rate, window_size=0.02, step_size=0.01, nfft=512, eps=1e-10):
    window_length = int(sample_rate*window_size)
    hop_length = int(sample_rate*step_size)

    freqs, time, spec = signal.spectrogram(data, fs=sample_rate, window='hann', nfft=nfft,  nperseg=window_length,
                                        noverlap=hop_length, detrend=False)
    log_spec = np.log10(spec.astype(np.float32)+eps)
    return freqs, time, log_spec


def convert2model(file):
    sr, data = wavfile.read(file)
    data = pre_emphesis(data)
    _, _, log_spectro = log_spectrogram(data, sample_rate=sr)
    # width, height = log_spectro.shape[0], log_spectro.shape[1]
    # convert shape(257,49) to shape(257, 50)
    log_spectro = np.resize(log_spectro, (1, 257, 50))
    return log_spectro


def plot_wav(data):
    plt.plot(data)
    plt.xlabel('time')
    plt.show()


if __name__ == '__main__':
    # files = Path('./val').glob('*wav')
    # for i, file in enumerate(files):
    #     if i>50:
    #         break
    #     data, sr = librosa.load(file, sr=None)
    #     # spec = librosa.feature.melspectrogram(data, sr=sr)
    #     librosa.display.waveplot(data, sr=sr, x_axis='time')
    #     plt.ylabel('amplitude')
    #     # librosa.display.specshow(spec, sr=sr,x_axis='time', y_axis='log')
    #     plt.gcf().savefig('./img/{}.png'.format(file.name))
    #     plt.clf()
    # data, sr = librosa.load('test1.wav', sr=None)
    sr, data = wavfile.read('test.wav')
    # # _, _, spec = log_spectrogram(data, sample_rate=sr)
    # spec = librosa.feature.melspectrogram(data, sr=sr)
    # log_spec = librosa.amplitude_to_db(spec, ref=np.max)
    # librosa.display.specshow(log_spec, x_axis='time', y_axis='log')
    #
    # plt.savefig('spec.png')
    _, _, a = log_spectrogram(data, sample_rate=sr)
    print(a.shape)
    # data = data[4000:4400]
    # # plot_wav(data)
    # # plt.figure(figsize=(12, 7))
    # # plt.subplot(121)
    # plt.title('Phổ tím hiệu')
    # plt.magnitude_spectrum(data, Fs=sr, scale='dB')
    #
    # # plt.subplot(122)
    # # plt.title('Qua cửa sổ ')
    # # # data = pre_emphesis(data)
    # # plt.magnitude_spectrum(data, Fs=sr, scale='dB', window=np.hanning(400))
    # # # plt.show()
    # plt.savefig('spectrum.png', dpi=60)

