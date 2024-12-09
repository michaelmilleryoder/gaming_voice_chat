from pyannote.audio import Pipeline
from keys import hugging_face_key
import torch
import torchaudio
import numpy as np


pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1",
                                    use_auth_token=hugging_face_key)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

pipeline.to(device)

def diarize(input, output):
    diarization = pipeline(input)
    waveform, sample_rate = torchaudio.load(input)

    parts = []

    for s in diarization.itersegments():
        speaker = (list(diarization.get_labels(s))[0])
        start = s.start
        end = s.end
        audio = waveform.T[int(start * sample_rate) : int(end * sample_rate)].T
        audio = {'array': np.array(audio[0]),
    'sampling_rate': sample_rate}
        parts.append((speaker, audio))
    
    transcription = [speaker + ': "' + pipe(audio)['text'] + '"' for speaker, audio in parts]


    with open(output, 'w', encoding='utf8') as f:
        f.write('\n'.join(transcription))