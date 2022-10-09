from scipy.io import wavfile
from sklearn.decomposition import PCA
import numpy as np
import cv2, os
import moviepy.editor as mp
from scipy.io import wavfile
import soundfile as sf

number = 20


def pca_frame(frame, n):

    def transfer_PCA(color_layer):
        pca = PCA(n_components=n)
        pca.fit(color_layer)
        trans_pca = pca.transform(color_layer)
        return pca, trans_pca

    blue, green, red = [x for x in cv2.split(frame)]
    #PCA fit
    layer_blue = transfer_PCA(blue)
    layer_green = transfer_PCA(green)
    layer_red = transfer_PCA(red)
    #reconstroct
    res_blue = layer_blue[0].inverse_transform(layer_blue[1])
    res_green = layer_green[0].inverse_transform(layer_green[1])
    res_red = layer_red[0].inverse_transform(layer_red[1])
    result = cv2.merge((res_blue,res_green,res_red))

    return result


def video2Frames(name, f):
    
    if not os.path.exists(f):
        os.mkdir(f)

    folder = f + '/'

    cap = cv2.VideoCapture(name)
    i = 0
    while True:    
        _, frame = cap.read()
        if frame is None or cv2.waitKey(1) == ord('q'):
            break
        res = pca_frame(frame, number)
        cv2.imwrite(folder + 'loli.' + str(i) + '.jpg', res)
        print(folder + 'loli.' + str(i) + '.jpg')
        i += 1

    cap.release()
    cv2.destroyAllWindows()

    return folder

#convert to video

def Frames2video(filename, f):

    folder = f + '/'

    fps = 30
    out = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'mp4v'), fps, (1280, 720))
    dir = [folder + name for name in os.listdir(folder)]
    dir.sort(key=lambda x: int(x[len(folder):].split('.')[1]))
    
    for x in dir:
        frame = cv2.imread(x)
        out.write(frame)
    out.release()

    return filename
#extract wav from mp4


def sound_processing(mp4, file_name):

    def extractWav(mp4, wav):
        my_clip = mp.VideoFileClip(mp4)
        my_clip.audio.write_audiofile(wav)
        return wav

    fs, wav = wavfile.read(extractWav(mp4, r"LoliDance.wav"))

    def reduce_dims(audio, n_components=number, size=512):
        hanging = 512 - np.mod(len(audio), 512)
        padded = np.pad(audio, (0, hanging), 'constant', constant_values=0)

        mus_arr = padded.reshape((len(padded) // 512, 512))

        pca = PCA(n_components=n_components)
        pca.fit(mus_arr)

        arr_trans = pca.transform(mus_arr)
        reconstruct = pca.inverse_transform(arr_trans).reshape((len(padded)))
        return reconstruct

    res = reduce_dims(wav[:,0]), reduce_dims(wav[:,1])
    audio = np.array(res).T

    sf.write(file_name, audio, fs)

    return file_name

#2 file wav and mp4(no sound) - merge them

def combine_audio(vidname, audname, outname, fps=30):
    my_clip = mp.VideoFileClip(vidname)
    audio_background = mp.AudioFileClip(audname)
    final_clip = my_clip.set_audio(audio_background)
    final_clip.write_videofile(outname,fps=fps)

vidname = Frames2video('LoliDancePCA.mp4', video2Frames('LoliDance.mp4', 'pca_frame'))
audname = sound_processing(r'LoliDance.mp4', r'LoliDance.wav')

outname = 'Result.mp4'
combine_audio(vidname, audname, outname)