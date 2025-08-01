# Copyright (C) 2021-22 Intel Corporation
# SPDX-License-Identifier: MIT
# See: https://spdx.org/licenses/

import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../microsoft_dns/")

import glob
import argparse
import configparser as CP

import librosa
import numpy as np
import pandas as pd
from scipy import signal
from scipy.io import wavfile
from uuid import uuid4

import random
from random import shuffle
from microsoft_dns.noisyspeech_synthesizer_singleprocess import add_pyreverb, build_audio, gen_audio
from microsoft_dns.audiolib import (
    audioread,
    audiowrite,
    normalize_segmental_rms,
    active_rms,
    EPS,
    activitydetector,
    is_clipped,
    add_clipping,
)
from microsoft_dns import utils

MAXTRIES = 50
MAXFILELEN = 100


np.random.seed(5)
random.seed(5)


def segmental_snr_mixer(params, clean, noise, snr, target_level=-25, clipping_threshold=0.99):
    """Function to mix clean speech and noise at various segmental SNR levels"""
    if len(clean) > len(noise):
        noise = np.append(noise, np.zeros(len(clean) - len(noise)))
    else:
        clean = np.append(clean, np.zeros(len(noise) - len(clean)))
    clean = clean / (max(abs(clean)) + EPS)
    noise = noise / (max(abs(noise)) + EPS)
    rmsclean, rmsnoise = active_rms(clean=clean, noise=noise)
    clean = normalize_segmental_rms(clean, rms=rmsclean, target_level=target_level)
    noise = normalize_segmental_rms(noise, rms=rmsnoise, target_level=target_level)
    # Set the noise level for a given SNR
    noisescalar = rmsclean / (10 ** (snr / 20)) / (rmsnoise + EPS)
    noisenewlevel = noise * noisescalar

    # Mix noise and clean speech
    noisyspeech = clean + noisenewlevel
    # Randomly select RMS value between -15 dBFS and -35 dBFS and normalize
    # noisyspeech with that value.
    # There is a chance of clipping that might happen with very less
    # probability, which is not a major issue.
    noisy_rms_level = np.random.randint(params["target_level_lower"], params["target_level_upper"])
    rmsnoisy = (noisyspeech**2).mean() ** 0.5
    scalarnoisy = 10 ** (noisy_rms_level / 20) / (rmsnoisy + EPS)
    noisyspeech = noisyspeech * scalarnoisy
    clean = clean * scalarnoisy
    noisenewlevel = noisenewlevel * scalarnoisy
    # Final check to see if there are any amplitudes exceeding +/- 1.
    # If so, normalize all the signals accordingly
    if is_clipped(noisyspeech):
        noisyspeech_maxamplevel = max(abs(noisyspeech)) / (clipping_threshold - EPS)
        noisyspeech = noisyspeech / noisyspeech_maxamplevel
        clean = clean / noisyspeech_maxamplevel
        noisenewlevel = noisenewlevel / noisyspeech_maxamplevel
        noisy_rms_level = int(20 * np.log10(scalarnoisy / noisyspeech_maxamplevel * (rmsnoisy + EPS)))

    return clean, noisenewlevel, noisyspeech, noisy_rms_level


def gen_audio_NECOTIS(file_queue, is_clean, params, audio_samples_length=-1):
    """Calls build_audio() to get an audio signal, and verify that it meets the
    activity threshold"""

    clipped_files = []
    low_activity_files = []
    if audio_samples_length == -1:
        audio_samples_length = int(params["audio_length"] * params["fs"])
    if is_clean:
        activity_threshold = params["clean_activity_threshold"]
    else:
        activity_threshold = params["noise_activity_threshold"]

    while True:
        audio, source_files, new_clipped_files = build_audio_NECOTIS(file_queue, is_clean, params, audio_samples_length)

        clipped_files += new_clipped_files
        if len(audio) < audio_samples_length:
            continue

        if activity_threshold == 0.0:
            break

        percactive = activitydetector(audio=audio)
        if percactive > activity_threshold:
            break
        else:
            low_activity_files += source_files

    return audio, source_files, clipped_files, low_activity_files

def build_audio_NECOTIS(file_queue, is_clean, params, audio_samples_length=-1):
    """Construct an audio signal from source files"""

    fs_output = params["fs"]
    silence_length = params["silence_length"]
    if audio_samples_length == -1:
        audio_samples_length = int(params["audio_length"] * params["fs"])

    output_audio = np.zeros(0)
    remaining_length = audio_samples_length
    files_used = []
    clipped_files = []

    # if is_clean:
    #     source_files = params["cleanfilenames"]
    #     idx = index
    # else:
    #     if "noisefilenames" in params.keys():
    #         source_files = params["noisefilenames"]
    #         idx = index
    #     # if noise files are organized into individual subdirectories, pick a directory randomly
    #     else:
    #         noisedirs = params["noisedirs"]
    #         # pick a noise category randomly
    #         idx_n_dir = np.random.randint(0, np.size(noisedirs))
    #         source_files = glob.glob(os.path.join(noisedirs[idx_n_dir], params["audioformat"]))
    #         shuffle(source_files)
    #         # pick a noise source file index randomly
    #         idx = np.random.randint(0, np.size(source_files))

    # initialize silence
    silence = np.zeros(int(fs_output * silence_length))

    # iterate through multiple clips until we have a long enough signal
    tries_left = MAXTRIES
    while remaining_length > 0 and tries_left > 0:
        # read next audio file and resample if necessary
        source_file = file_queue.get()
        #idx = (idx + 1) % np.size(source_files)
        input_audio, fs_input = audioread(source_file)
        if input_audio is None:
            sys.stderr.write("WARNING: Cannot read file: %s\n" % source_file)
            continue
        if fs_input != fs_output:
            input_audio = librosa.resample(input_audio, orig_sr=fs_input, target_sr=fs_output)

        # if current file is longer than remaining desired length, and this is
        # noise generation or this is training set, subsample it randomly
        if len(input_audio) > remaining_length and (not is_clean or not params["is_test_set"]):
            idx_seg = np.random.randint(0, len(input_audio) - remaining_length)
            input_audio = input_audio[idx_seg : idx_seg + remaining_length]

        # check for clipping, and if found move onto next file
        if is_clipped(input_audio):
            clipped_files.append(source_file)
            tries_left -= 1
            continue

        # concatenate current input audio to output audio stream
        files_used.append(source_file)
        output_audio = np.append(output_audio, input_audio)
        remaining_length -= len(input_audio)

        # add some silence if we have not reached desired audio length
        if remaining_length > 0:
            silence_len = min(remaining_length, len(silence))
            output_audio = np.append(output_audio, silence[:silence_len])
            remaining_length -= silence_len

    if tries_left == 0 and not is_clean and "noisedirs" in params.keys():
        print(
            "There are not enough non-clipped files in the noise "
            + " directory to complete the audio build"
        )
        return [], [], clipped_files

    return output_audio, files_used, clipped_files




def _process(args):
    file_num, (clean_queue, noise_queue) = args
    clean_source_files = []
    clean_clipped_files = []
    clean_low_activity_files = []
    noise_source_files = []
    noise_clipped_files = []
    noise_low_activity_files = []

    while not (clean_queue.empty() or noise_queue.empty()): # Retry until file is done or queues are empty
        try:
            # generate clean speech
            clean, clean_sf, clean_cf, clean_laf = gen_audio_NECOTIS(
                clean_queue,
                is_clean=True, params=params, 
            )
            # generate noise
            noise, noise_sf, noise_cf, noise_laf = gen_audio_NECOTIS(
                noise_queue,
                is_clean=False, params=params, audio_samples_length=len(clean)
            )
            clean_clipped_files += clean_cf
            clean_low_activity_files += clean_laf
            noise_clipped_files += noise_cf
            noise_low_activity_files += noise_laf

            snr = np.random.randint(params["snr_lower"], params["snr_upper"])

            clean_snr, noise_snr, noisy_snr, target_level = segmental_snr_mixer(
                params=params, clean=clean, noise=noise, snr=snr
            )
        except ValueError as e:
            print("Found exception")
            print(str(e))
            print("Trying again")
            continue

        # unexpected clipping
        if is_clipped(clean_snr) or is_clipped(noise_snr) or is_clipped(noisy_snr):
            print(
                "Warning: File #"
                + str(file_num)
                + " has unexpected clipping, "
                + "returning without writing audio to disk"
            )
            continue

        clean_source_files += clean_sf
        noise_source_files += noise_sf

        # write resultant audio streams to files
        hyphen = "-"
        clean_source_filenamesonly = [i[:-4].split(os.path.sep)[-1] for i in clean_sf]
        clean_files_joined = hyphen.join(clean_source_filenamesonly)[:MAXFILELEN]
        noise_source_filenamesonly = [i[:-4].split(os.path.sep)[-1] for i in noise_sf]
        noise_files_joined = hyphen.join(noise_source_filenamesonly)[:MAXFILELEN]

        noisyfilename = (
            clean_files_joined
            + "_"
            + noise_files_joined
            + "_snr"
            + str(snr)
            + "_tl"
            + str(target_level)
            + "_fileid_"
            + str(file_num)
            + ".wav"
        )
        cleanfilename = "clean_fileid_" + str(file_num) + ".wav"
        noisefilename = "noise_fileid_" + str(file_num) + ".wav"

        dir = "validation_set/" if params["is_test_set"] else "training_set/"
        noisypath = os.path.join(params["root"] + dir + params["noisy_speech_dir"], noisyfilename)
        cleanpath = os.path.join(params["root"] + dir + params["clean_speech_dir"], cleanfilename)
        noisepath = os.path.join(params["root"] + dir + params["noise_dir"], noisefilename)

        audio_signals = [noisy_snr, clean_snr, noise_snr]
        file_paths = [noisypath, cleanpath, noisepath]

        for i in range(len(audio_signals)):
            try:
                audiowrite(file_paths[i], audio_signals[i], params["fs"])
            except Exception as e:
                print(str(e))

        return # File done



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-fs", type=int, default=16000, help="sampling frequency")
    parser.add_argument("-root", type=str, default="./", help="root dataset directory")
    parser.add_argument("-audio_length", type=float, default=30, help="minimum length of each audio clip (s)")
    parser.add_argument(
        "-silence_length",
        type=float,
        default=0.2,
        help="duration of silence introduced between clean speech utterances (s)",
    )
    parser.add_argument("-total_hours", type=float, default=500, help="total number of hours of data required (hrs)")
    parser.add_argument("-snr_lower", type=int, default=-5, help="lower bound of SNR required (dB)")
    parser.add_argument("-snr_upper", type=int, default=20, help="upper bound of SNR required (dB)")
    parser.add_argument("-target_level_lower", type=float, default=-35, help="lower bound of target audio (dB)")
    parser.add_argument("-target_level_upper", type=float, default=-15, help="upper bound of target audio (dB)")
    parser.add_argument("-target_snr_levels", type=int, default=21, help="total number of snr levels")
    parser.add_argument(
        "-clean_activity_threshold", type=float, default=0.6, help="activity threshold for clean speech"
    )
    parser.add_argument("-noise_activity_threshold", type=float, default=0.0, help="activity threshold for noise")
    parser.add_argument("-is_validation_set", type=bool, default=False, help="validation data flag")
    # parser.add_argument('-use_singing_data', type=bool, default=False, help='use singing data')
    # parser.add_argument('-use_emotion_data', type=bool, default=False, help='use emotion data')
    # parser.add_argument('-use_mandarin_data', type=bool, default=False, help='use mandarin data')
    # parser.add_argument('-reverb_table', type=str, default='RIR_table_simple.csv', help='reverberation table data')
    # parser.add_argument('-lower_t60', type=float, default=0.3, help='lower bound of t60 range in seconds')
    # parser.add_argument('-upper_t60', type=float, default=1.3, help='upper bound of t60 range in seconds')
    parser.add_argument("-noisy_speech_dir", type=str, default="noisy", help="noisy speech directory")
    parser.add_argument("-clean_speech_dir", type=str, default="clean", help="clean speech directory")
    parser.add_argument("-noise_dir", type=str, default="noise", help="noise directory")
    parser.add_argument("-log_dir", type=str, default="log", help="log directory")
    parser.add_argument("-fileindex_start", type=int, default=0, help="start file idx")

    params = vars(parser.parse_args())

    root = params["root"]
    params["is_test_set"] = params["is_validation_set"]

    params["num_files"] = int(params["total_hours"] * 3600 / params["audio_length"])
    # params['fileindex_start'] = 0
    params["fileindex_end"] = params["fileindex_start"] + params["num_files"] - 1
    print("Number of files to be synthesized:", params["num_files"])
    #print("Start idx:", params["fileindex_start"])
    #print("Stop idx:", params["fileindex_end"])
    print(f"Generating synthesized data in {root}")

    clean_dir = root + "datasets_fullband/clean_fullband"
    noise_dir = root + "datasets_fullband/noise_fullband"
    clean_filenames = glob.glob(clean_dir + "/**/*.wav", recursive=True)
    noise_filenames = glob.glob(noise_dir + "/**/*.wav", recursive=True)

    shuffle(clean_filenames)
    shuffle(noise_filenames)

    params["cleanfilenames"] = clean_filenames
    params["num_cleanfiles"] = len(clean_filenames)

    params["noisefilenames"] = noise_filenames
    params["num_noisefiles"] = len(noise_filenames)

    assert len(clean_filenames) > 0
    assert len(noise_filenames) > 0


    import multiprocessing
    from itertools import repeat

    from tqdm import tqdm
    manager = multiprocessing.Manager()
    clean_queue = manager.Queue()
    noise_queue = manager.Queue()

    for c in clean_filenames:
        clean_queue.put(c)
    for n in noise_filenames:
        noise_queue.put(n)

    print("Starting generation")
    
    with multiprocessing.Pool(12) as p:
        r = list(tqdm(p.imap_unordered(_process, enumerate(repeat((clean_queue, noise_queue), params["num_files"]))), total=params["num_files"]))

    exit(0)


    # Call synthesize() to generate audio
    # (
    #     clean_source_files,
    #     clean_clipped_files,
    #     clean_low_activity_files,
    #     noise_source_files,
    #     noise_clipped_files,
    #     noise_low_activity_files,
    # ) = synthesize(params)

    # Create log directory if needed, and write log files of clipped and low activity files
    log_dir = params["log_dir"] + "/"
    os.makedirs(log_dir, exist_ok=True)

    utils.write_log_file(log_dir, "source_files.csv", clean_source_files + noise_source_files)
    utils.write_log_file(log_dir, "clipped_files.csv", clean_clipped_files + noise_clipped_files)
    utils.write_log_file(log_dir, "low_activity_files.csv", clean_low_activity_files + noise_low_activity_files)

    # Compute and print stats about percentange of clipped and low activity files
    total_clean = len(clean_source_files) + len(clean_clipped_files) + len(clean_low_activity_files)
    total_noise = len(noise_source_files) + len(noise_clipped_files) + len(noise_low_activity_files)
    pct_clean_clipped = round(len(clean_clipped_files) / total_clean * 100, 1)
    pct_noise_clipped = round(len(noise_clipped_files) / total_noise * 100, 1)
    pct_clean_low_activity = round(len(clean_low_activity_files) / total_clean * 100, 1)
    pct_noise_low_activity = round(len(noise_low_activity_files) / total_noise * 100, 1)

    print(
        "Of the "
        + str(total_clean)
        + " clean speech files analyzed, "
        + str(pct_clean_clipped)
        + "% had clipping, and "
        + str(pct_clean_low_activity)
        + "% had low activity "
        + "(below "
        + str(params["clean_activity_threshold"] * 100)
        + "% active percentage)"
    )
    print(
        "Of the "
        + str(total_noise)
        + " noise files analyzed, "
        + str(pct_noise_clipped)
        + "% had clipping, and "
        + str(pct_noise_low_activity)
        + "% had low activity "
        + "(below "
        + str(params["noise_activity_threshold"] * 100)
        + "% active percentage)"
    )
