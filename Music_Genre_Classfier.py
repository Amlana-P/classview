import librosa.display
import matplotlib.pyplot as plt
import sklearn

# Loading the audio file in wav format
audio_path = 'C:/Users/Hp/PycharmProjects/Classview.tech/music.wav'
x, sr = librosa.load(audio_path)

# Plotting the waveform of the audio file
plt.figure(figsize=(14, 5))
librosa.display.waveplot(x, sr=sr)

# Spectrogram of Audio file
X = librosa.stft(x)
Xdb = librosa.amplitude_to_db(abs(X))
plt.figure(figsize=(14, 5))
librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='hz')
plt.colorbar()

# Computing Zero-Crossings
n0 = 9000
n1 = 9100
zero_crossings = librosa.zero_crossings(x[n0:n1], pad=False)
print(sum(zero_crossings))

# Computing Spectral-Centroid
spectral_centroids = librosa.feature.spectral_centroid(x, sr=sr)[0]

# Computing the time variable for visualization
frames = range(len(spectral_centroids))
t = librosa.frames_to_time(frames)


# Normalising the spectral centroid for visualisation
def normalize(x, axis=0):
    return sklearn.preprocessing.minmax_scale(x, axis=axis)


# Plotting the Spectral Centroid along the waveform
librosa.display.waveplot(x, sr=sr, alpha=0.4)
plt.figure(figsize=(14, 5))
plt.plot(t, normalize(spectral_centroids), color='r')

# Computing Spectral Rolloff
spectral_rolloff = librosa.feature.spectral_rolloff(x+0.01, sr=sr)[0]
librosa.display.waveplot(x, sr=sr, alpha=0.4)
plt.plot(t, normalize(spectral_rolloff), color='g')

# Computing & Displaying  the MFCCs:
mfccs = librosa.feature.mfcc(x, sr=sr)
plt.figure(figsize=(14, 5))
librosa.display.specshow(mfccs, sr=sr, x_axis='time')

# Computing Chroma Frequencies
hop_length = 512
chromagram = librosa.feature.chroma_stft(x, sr=sr, hop_length=hop_length)
plt.figure(figsize=(15, 5))
librosa.display.specshow(chromagram, x_axis='time', y_axis='chroma', hop_length=hop_length, cmap='coolwarm')

plt.show()
