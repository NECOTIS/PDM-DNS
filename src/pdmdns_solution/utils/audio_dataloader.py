import os
import glob
import torch
import numpy as np
import re
from typing import Tuple, Dict, Any
from scipy.io.wavfile import read
from torch.utils.data import Dataset
import h5py
import librosa
import random
from intel_code.noisyspeech_synthesizer import segmental_snr_mixer, is_clipped, activitydetector


class DNSAudio:
    """Aduio dataset loader for DNS.

    Parameters
    ----------
    root : str, optional
        Path of the dataset location, by default './'.
    """

    def __init__(self, root: str = "./", normalize=False, audio_length=30) -> None:
        self.root = root
        # TODO : resolve files missing
        self.noisy_files = glob.glob(root + "noisy/**.wav")
        self.file_id_from_name = re.compile("fileid_(\d+)")
        self.snr_from_name = re.compile("snr(-?\d+)")
        self.target_level_from_name = re.compile("tl(-?\d+)")
        self.source_info_from_name = re.compile("^(.*?)_snr")
        self.normalize = normalize
        self.audio_length = audio_length # in seconds

    def _get_filenames(self, n: int) -> Tuple[str, str, str, Dict[str, Any]]:
        noisy_file = self.noisy_files[n % self.__len__()]
        filename = noisy_file.split(os.sep)[-1]
        file_id = int(self.file_id_from_name.findall(filename)[0])
        clean_file = self.root + f"clean/clean_fileid_{file_id}.wav"
        noise_file = self.root + f"noise/noise_fileid_{file_id}.wav"
        snr = int(self.snr_from_name.findall(filename)[0])
        target_level = int(self.target_level_from_name.findall(filename)[0])
        source_info = self.source_info_from_name.findall(filename)[0]
        metadata = {"snr": snr, "target_level": target_level, "source_info": source_info}
        return noisy_file, clean_file, noise_file, metadata

    def __getitem__(self, n: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        """Gets the nth sample from the dataset.

        Parameters
        ----------
        n : int
            Index of the dataset sample.

        Returns
        -------
        np.ndarray
            Noisy audio sample.
        np.ndarray
            Clean audio sample.
        np.ndarray
            Noise audio sample.
        Dict
            Sample metadata.
        """
        noisy_file, clean_file, noise_file, metadata = self._get_filenames(n)
        try:
            sampling_frequency, noisy_audio = read(noisy_file)
            _, clean_audio = read(clean_file)
            _, noise_audio = read(noise_file)
        except Exception as e:
            print(f"COULDN'T READ FILE {n}")
            os.remove(noisy_file)

            print(e)

            noisy_file, clean_file, noise_file, metadata = self._get_filenames(n + 1)

            sampling_frequency, noisy_audio = read(noisy_file)
            _, clean_audio = read(clean_file)
            _, noise_audio = read(noise_file)

        num_samples = self.audio_length * sampling_frequency
        metadata["fs"] = sampling_frequency

        if len(noisy_audio) > num_samples:
            noisy_audio = noisy_audio[:num_samples]
        else:
            noisy_audio = np.concatenate([noisy_audio, np.zeros(num_samples - len(noisy_audio))])
        if len(clean_audio) > num_samples:
            clean_audio = clean_audio[:num_samples]
        else:
            clean_audio = np.concatenate([clean_audio, np.zeros(num_samples - len(clean_audio))])
        if len(noise_audio) > num_samples:
            noise_audio = noise_audio[:num_samples]
        else:
            noise_audio = np.concatenate([noise_audio, np.zeros(num_samples - len(noise_audio))])

        x = 2**15 if self.normalize else 1
        noisy_audio = noisy_audio.astype(np.float32) / x
        clean_audio = clean_audio.astype(np.float32) / x
        noise_audio = noise_audio.astype(np.float32) / x
        self.last_metadata = metadata
        return noisy_audio, clean_audio, noise_audio

    def __len__(self) -> int:
        """Length of the dataset."""
        return len(self.noisy_files)







class H5DNSAudio(Dataset):
    def __init__(self, root: str, is_validation=False, seed=0x1B, fs=16000, audio_length=30, silence_length=0.2, activity_threshold=False, snr_lower=-5, max_nb_files=10000) -> None:
        """Generator of noisy audio based on MS DNS subset file

        Args:
            root (str): path of hdf5 file generated
            seed (int, optional): random seed. Defaults to 0x1B.
            fs (int, optional): sampling frequency. Defaults to 16000.
            audio_length (float, optional): minimum length of each audio clip (s). Defaults to 30.
            silence_length (float, optional): duration of silence introduced between clean speech utterances (s). Defaults to 0.2.
        """
        super().__init__()
        self.root = root
        self.is_validation = is_validation
        self.subdir = "validation" if self.is_validation else "training"
        self.set_seed(seed)
        self.max_nb_files = max_nb_files

        self.activity_threshold = activity_threshold
        self.fs = fs
        self.audio_length = audio_length
        self.audio_n_samples = int(audio_length * fs)
        self.silence_length = silence_length
        self.silence_n_samples = int(silence_length * fs)
        self.snr_lower = snr_lower
        self.snr_params = dict(snr_lower=self.snr_lower, snr_upper=20, target_level_lower=-35, target_level_upper=-15)
        self.all_snr = self.rnd.randint(self.snr_params["snr_lower"], self.snr_params["snr_upper"], size=self.max_nb_files) if self.is_validation else None
        
        self.clean_file_paths, self.noise_file_paths = self._get_h5_file_paths()
        self.clean_file_data = self._preload_h5_files(self.clean_file_paths)
        self.noise_file_data = self._preload_h5_files(self.noise_file_paths)
        print(len(self.clean_file_data), "Clean data ", len(self.noise_file_data), "Noise data")
    
    
    def _get_h5_file_paths(self):
        """
        Get all the file paths of the H5 files in the directory.
        File names follow the pattern "DNSDataset{_val if is_validation else ''}_{batch_count}.h5".
        """
        # Define the root directory and subdirectory
        dirr = os.path.join(self.root, self.subdir)
        
        # Using glob to get all relevant files with the pattern matching .h5 files
        clean_file_paths = glob.glob(os.path.join(dirr, "DNSDataset_clean*.h5"))
        noise_file_paths = glob.glob(os.path.join(dirr, "DNSDataset_noise*.h5"))
        
        # Function to filter valid file paths by attempting to open them
        def filter_valid_files(file_paths):
            valid_files = []
            for file_path in file_paths:
                try:
                    with h5py.File(file_path, 'r') as f:
                        pass  # Simply open the file to check if it is valid
                    valid_files.append(file_path)  # If no exception, add to valid list
                except: 
                    pass  # Skip invalid file
            return valid_files
        
        # Filter valid clean and noise file paths
        clean_file_paths = filter_valid_files(clean_file_paths)
        noise_file_paths = filter_valid_files(noise_file_paths)
        # Assert that the lists are not empty
        assert clean_file_paths, f"No valid 'clean' files found in {dirr}"
        assert noise_file_paths, f"No valid 'noise' files found in {dirr}"
        return sorted(clean_file_paths), sorted(noise_file_paths)  # Sort to ensure proper batching order
    
    def _preload_h5_files(self, file_paths):
        """
        Pre-load all the HDF5 files, excluding 'scheme' and 'sample_rate'.
        For each file, store the min/max indices and the dataset keys.
        """
        file_data = []
        
        for file_path_id,file_path in enumerate(file_paths):
            with h5py.File(file_path, 'r') as f:
                file_data += [(key,file_path_id) for key in f["data"].keys()]
        random.shuffle(file_data)
        return file_data
    
    def _load_data(self, file_data, file_paths, selected_id):
        # key, file_id = self.rnd.choice(file_data)
        key, file_id = file_data[selected_id]
        with h5py.File(file_paths[file_id], 'r') as f:
            sampling_frequency = f['sample_rate'][()]
            audio = f['data'][key][:].astype(np.float32) / 2**15
        return audio, sampling_frequency


    def set_seed(self, seed):
        self.id = seed
        self.rnd = np.random.RandomState(seed=seed)
        random.seed(seed)


    def play_audio(self, audio):
        import simpleaudio as sa

        play_obj = sa.play_buffer(audio.astype(np.float32), 1, 4, self.fs)
        play_obj.wait_done()

    def build_audio(self, file_data, file_paths, thresh, idx):
        silence = np.zeros(int(self.silence_n_samples), dtype=np.float32)
        remaining_length = self.audio_n_samples
        output_audio = np.zeros(0, dtype=np.float32)
        
        # Ensure reproducibility over epochs for validation set
        # For train set, data will differs over epochs
        if idx is not None: rnd = np.random.RandomState(seed=idx)
        while remaining_length > 0:
            selected_id = self.rnd.randint(0, len(file_data)) if idx is None else rnd.randint(0, len(file_data))
            sound_ndarray, sound_fs = self._load_data(file_data, file_paths, selected_id)

            if sound_fs != self.fs:
                sound_ndarray = librosa.resample(sound_ndarray, orig_sr=sound_fs, target_sr=self.fs)

            if len(sound_ndarray) > remaining_length:
                start_ = np.random.randint(0,len(sound_ndarray) - remaining_length)
                sound_ndarray = sound_ndarray[start_:start_+remaining_length]

            if is_clipped(sound_ndarray):
                continue

            output_audio = np.append(output_audio, sound_ndarray)
            remaining_length -= len(sound_ndarray)

            if remaining_length > 0:
                silence_len = min(remaining_length, len(silence))
                output_audio = np.append(output_audio, silence[:silence_len])
                remaining_length -= silence_len
            if self.activity_threshold and remaining_length <= 0:
                percactive = activitydetector(audio=output_audio)
                if percactive < thresh:
                    # Start over
                    output_audio = np.zeros(0, dtype=np.float32)
                    remaining_length = self.audio_n_samples

        return output_audio


    def __len__(self):
        return self.max_nb_files
    
    def __getitem__(self, idx):
        clean = self.build_audio(self.clean_file_data, self.clean_file_paths, 0.6, idx if self.is_validation else None)
        noise = self.build_audio(self.noise_file_data, self.noise_file_paths, 0.0, idx if self.is_validation else None)
        snr = self.all_snr[idx] if self.is_validation else self.rnd.randint(self.snr_params["snr_lower"], self.snr_params["snr_upper"])
        clean_snr, noise_snr, noisy_snr, target_level = segmental_snr_mixer(
            params=self.snr_params, clean=clean, noise=noise, snr=snr
        )

        return clean_snr.astype(np.float32), noise_snr.astype(np.float32), noisy_snr.astype(np.float32), target_level
        


if __name__ == "__main__":
    dataset = H5DNSAudio(root="datasets/output_all_h5", is_validation=False, seed=0x1B, fs=16000, audio_length=5)
    print(len(dataset))
    for clean, noise, noisy, target_level in dataset:
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        axs[0].plot(clean)
        axs[0].set_title('Clean')
        axs[1].plot(noise)
        axs[1].set_title('Noise')
        axs[2].plot(noisy)
        axs[2].set_title('Noisy')
        plt.tight_layout()
        plt.show()
        break

    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=2,
        #num_workers=2,
        pin_memory=False,
    )
    data = next(iter(train_loader))
    for clean, noise, noisy in zip(data[0],data[1],data[2]):
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        axs[0].plot(clean)
        axs[0].set_title('Clean')
        axs[1].plot(noise)
        axs[1].set_title('Noise')
        axs[2].plot(noisy)
        axs[2].set_title('Noisy')
        plt.tight_layout()
        plt.show()

    
    





























