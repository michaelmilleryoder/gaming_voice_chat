########################################################################################
# IMPORTS
########################################################################################
from pyannote.audio import Pipeline
import torch
import torchaudio
import numpy as np
import librosa as rs
import json
import soundfile as sf
import tempfile
import copy
from audiotsm import wsola
from audiotsm.io.wav import WavReader, WavWriter
import os
import scipy
from scipy import signal
from scipy.io import wavfile
from scipy.signal import resample, lfilter
import optuna as op
from joblib import Parallel, delayed
from functools import partial
from pathlib import Path
import shutil
from tqdm import tqdm
from transformers import pipeline as hf_pipeline
import argparse
import moviepy

########################################################################################
########################################################################################

########################################################################################
# MODEL SETUP
########################################################################################
hf_key = os.getenv("HF_TOKEN")

# First loading in the diarization model
pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1", use_auth_token=hf_key
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pipeline.to(device)

# Additionally loading in this -- not initially in the diarize.py, but
# is needed.
pipe = hf_pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-base",
    generate_kwargs={"language": "english"},
)
########################################################################################
########################################################################################

########################################################################################
# PROCESSING FUNCTIONS
########################################################################################


# Voice anonymization largely taken from: https://github.com/sarulab-speech/lightweight_spkr_anon/tree/master
def diarize(wav_in_path, txt_out_path):
    diarization = pipeline(wav_in_path)
    waveform, sample_rate = torchaudio.load(wav_in_path)

    parts = []

    for s in diarization.itersegments():
        speaker = list(diarization.get_labels(s))[0]
        start = s.start
        end = s.end
        audio = waveform.T[int(start * sample_rate) : int(end * sample_rate)].T
        audio = {"array": np.array(audio[0]), "sampling_rate": sample_rate}
        parts.append((speaker, audio))

    transcription = [
        speaker + ': "' + pipe(audio)["text"] + '"' for speaker, audio in parts
    ]

    with open(txt_out_path, "w", encoding="utf8") as f:
        f.write("\n".join(transcription))


def resampling(x, coef=1.0, fs=16000):
    with tempfile.NamedTemporaryFile(mode="r", suffix=".wav", delete=True) as fn_r:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".wav", delete=True) as fn_w:
            sf.write(fn_r.name, x, fs, "PCM_16")
            with WavReader(fn_r.name) as fr:
                with WavWriter(fn_w.name, fr.channels, fr.samplerate) as fw:
                    tsm = wsola(
                        channels=fr.channels,
                        speed=coef,
                        frame_length=256,
                        synthesis_hop=int(fr.samplerate / 70.0),
                    )
                    tsm.run(fr, fw)

            y = resample(rs.load(fn_w.name)[0], len(x)).astype(x.dtype)
            fn_r.close()
            fn_w.close()
    return y


def anonymize(x, fs=16000, resamp_param=1.168):
    """anonymize speech using model parameters

    Args:
    x (list): waveform data
    fs (int): sampling frequency [Hz]
    kargs (dic): model parameters
    Returns:
    y (list): anonymized waveform data
    """

    y = copy.deepcopy(x)
    y = resampling(y, resamp_param, fs=fs)
    return y


def anonymize_wav(wav_in_path, wav_out_path):
    """
    A function that runs the voice anonimization on the input WAV file, saving
    the anonimized version to the out path.
    """

    fs = 16000  # sampling frequency

    # load wav
    x = rs.load(wav_in_path, sr=fs)[0]

    # anonymize using vctk male params
    y = anonymize(x, fs, resamp_param=1.168)

    # save wav
    sf.write(wav_out_path, y, fs, "PCM_16")


def nonintegrated_diarization_anonymization(
    wav_in_path, wav_out_path, txt_out_path, pbar
):
    """
    A non-integrated form of the diarization/anonymization where they run
    sequentially on the entire video, rather than using the speaker segments
    generated with the diarization.
    """

    pbar.set_postfix_str("Diarizing and transcribing audio...")
    diarize(wav_in_path, txt_out_path)
    pbar.set_postfix_str("Anonymizing audio...")
    anonymize_wav(wav_in_path, wav_out_path)


def integrated_diarization_anonymization(wav_in_path, wav_out_path, txt_out_path, pbar):
    """
    A function that runs both the diarization and anonymization on the same video. The
    anonymization is only run on the snippets of speech extracted from the diarization
    procedure, before being concatenated together.
    """

    # Comments are to the best of my understanding

    #####
    pbar.set_postfix_str("Diarizing audio...")
    #####

    # Run diarization to get speaker labels and audio segments -- additionally
    # load in the waveform itself and the sample_rate from torchaudio.
    diarization = pipeline(wav_in_path)
    waveform, sample_rate = torchaudio.load(wav_in_path)

    parts = []
    all_audio_parts = []

    # Going through all spwaker segments
    for s in diarization.itersegments():

        # Getting the label and the timestamp start/end for that speajer
        speaker = list(diarization.get_labels(s))[0]
        start = s.start
        end = s.end

        # We then truncate the audio tensor to be just that interval
        audio = waveform.T[int(start * sample_rate) : int(end * sample_rate)].T
        all_audio_parts.append(audio)
        audio = {"array": np.array(audio[0]), "sampling_rate": sample_rate}

        # And append the truncated tensor and speaker label to parts
        parts.append((speaker, audio))

    #####
    pbar.set_postfix_str("Anonymizing speaking segments of audio...")
    #####

    # Now that we have done the actual diarization, we can now run the
    # anonimization of the audio. For this, we assume all segment produced
    # from above have the same sampling rate.
    compiled_audio_snippets = torch.cat([x for x in all_audio_parts], dim=1)

    # We save this as an aggregate of these un-anonymized snippets
    torchaudio.save("./aggregated_snippets.wav", compiled_audio_snippets, sample_rate)

    # We then use this path as what we anonymize
    anonymize_wav("./aggregated_snippets.wav", wav_out_path)

    # And then we delete the aggregated_snippets
    os.remove("./aggregated_snippets.wav")

    #####
    pbar.set_postfix_str("Transcribing audio...")
    #####

    # Our transcription is then going through each audio segment, getting the
    # transcription using Whisper, and then saving that to a list with the
    # label.
    transcription = [
        speaker + ': "' + pipe(audio)["text"] + '"' for speaker, audio in parts
    ]

    # Then save to a text file.
    with open(txt_out_path, "w", encoding="utf8") as f:
        f.write("\n".join(transcription))


########################################################################################
########################################################################################


########################################################################################
# PIPELINE
########################################################################################


def run_pipeline(
    in_folder_path,
    out_folder_path,
    accepted_file_formats=[
        ".wav",
        ".mp4",
    ],
    diarization_anonymization=nonintegrated_diarization_anonymization,
):
    """
    A function that runs the pipeline through every video within the folder_path.
    """

    print(
        "Beginning to run post-processing pipeline. Using all files with accepted format in:\n"
        + in_folder_path
        + "\n"
    )

    # We go through every filename within the folder
    pbar = tqdm(os.listdir(in_folder_path))
    for filename in pbar:

        # If it has one of the accepted file extensions, we proceed
        curr_file_ext = os.path.splitext(filename)[1]
        if curr_file_ext in accepted_file_formats:

            pbar.set_description_str(f"Post-processing file: {filename}")

            # If it is mp4, we need to save it as a .WAV
            if curr_file_ext == ".mp4":

                video_clip = moviepy.VideoFileClip(in_folder_path + filename)
                video_clip.audio.write_audiofile(
                    in_folder_path + filename[:-4] + ".wav", codec="pcm_s16le"
                )
                video_clip.close()

            filename_no_extension = filename[:-4]

            # We get the full path of that file
            wav_in_path = in_folder_path + filename_no_extension + ".wav"

            # And with that, we can get the output names for our anonimized
            # audio and text diarization.
            wav_out_path = out_folder_path + f"anonymized_{filename_no_extension}.wav"
            txt_out_path = (
                out_folder_path + f"diarized_transcript_{filename_no_extension}.txt"
            )

            # We then try to diarize/anonimize -- if it fails, we just
            # continue on to the next one.
            try:
                diarization_anonymization(wav_in_path, wav_out_path, txt_out_path, pbar)
                pbar.set_postfix_str("Done! On to the next...")

            except Exception as e:
                print(e)
                print("Failed to diarize and anonymize this file. Moving on.")
                continue


########################################################################################
########################################################################################


def main():
    parser = argparse.ArgumentParser(description="Process folders.")
    parser.add_argument(
        "in_folder_path",
        type=str,
        help="Path to the input folder, housing un-processed videos.",
    )
    parser.add_argument(
        "out_folder_path",
        type=str,
        help="Path to the output folder where processed video, audio, and text will be saved.",
    )

    args = parser.parse_args()
    run_pipeline(args.in_folder_path, args.out_folder_path)


if __name__ == "__main__":
    main()
