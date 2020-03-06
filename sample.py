import pyaudio
import wave
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
import random
import time

FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 1024
RECORD_SECONDS = 2
WAVE_OUTPUT_FILENAME = "file.wav"


def record():
    # start Recording

    stream = audio.open(format=FORMAT, channels=CHANNELS,
                        rate=RATE, input=True,
                        output=True,
                        frames_per_buffer=CHUNK)

    label = random.randrange(0, 10)
    print(label)
    frames = []

    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        print(len(data))
        frames.append(data)
    # print("finished recording")

    # stop Recording
    stream.stop_stream()
    stream.close()
    audio.terminate()

    waveFile = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    waveFile.setnchannels(CHANNELS)
    waveFile.setsampwidth(audio.get_sample_size(FORMAT))
    waveFile.setframerate(RATE)
    waveFile.writeframes(b''.join(frames))
    waveFile.close()
    return label


def plot_signal(signal):
    # plt.plot(signal)
    # plt.show()
    fig = plt.figure()
    plt.plot(signal)
    # plt.show()
    plt.draw()
    plt.pause(0.1)  # <-------
    input("<Hit Enter To Close>")
    plt.close(fig)


def split_signal(signal, alpha=0.4):
    beta = alpha+0.5
    return signal[int(alpha*RATE):int(beta*RATE)]


if __name__ == '__main__':

    while(True):
        audio = pyaudio.PyAudio()
        label = record()
        sr, signal = wav.read(WAVE_OUTPUT_FILENAME)
        plot_signal(signal)
        time.sleep(1)
        while(True):
            a = input('Choose alpha: ')
            new_signal = split_signal(signal, float(a))
            plot_signal(new_signal)
            ok = input('OK?:')
            if ok == '1':
                xxxx = random.randrange(10000, 99999)
                wav.write(filename='./data/{}_{}.wav'.format(label, xxxx), rate=RATE,
                          data=new_signal)
                # plt.gcf().savefig('./img/{}_{}.png'.format(label, xxxx), dpi=100)
                print('saved {}_{}.wav'.format(label, xxxx))
                break
            elif ok =='0':
                break

        t = input('Co muon thu tiep k?')
        if t=='0':
            break





