import os
import glob
import torch
import numpy as np
import re
import soundfile as sf
from typing import Tuple, Dict, Any
from scipy.io.wavfile import read
import json

class DNSAudio:
    """Aduio dataset loader for DNS.

    Parameters
    ----------
    root : str, optional
        Path of the dataset location, by default './'.
    """

    def __init__(self, root: str = "./", normalize=False) -> None:
        self.root = root
        self.noisy_files = glob.glob(os.path.join(root, "noisy/**.wav")) #glob.glob(root + "noisy/**.wav")
        self.noisy_files = [os.path.normpath(f) for f in self.noisy_files]
        self.file_id_from_name = re.compile("fileid_(\d+)")
        # self.snr_from_name = re.compile("snr(-?\d+)")
        # self.target_level_from_name = re.compile("tl(-?\d+)")
        # self.source_info_from_name = re.compile("^(.*?)_snr")
        self.normalize = normalize
        with open(os.path.join(root, "file_metadata.json")) as json_file:
            self.metadata = json.load(json_file)

    def _get_filenames(self, n: int) -> Tuple[str, str, str, Dict[str, Any]]:
        noisy_file = self.noisy_files[n % self.__len__()]
        filename = noisy_file.split(os.sep)[-1]
        file_id = int(self.file_id_from_name.findall(filename)[0])
        clean_file = os.path.join(self.root, f"clean/clean_fileid_{file_id}.wav") 
        noise_file = os.path.join(self.root, f"noise/noise_fileid_{file_id}.wav")  
        # snr = int(self.snr_from_name.findall(filename)[0])
        # target_level = int(self.target_level_from_name.findall(filename)[0])
        # source_info = self.source_info_from_name.findall(filename)[0]
        # metadata = {"snr": snr, "target_level": target_level, "source_info": source_info}
        metadata = self.metadata[str(file_id)]
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
            # os.remove(noisy_file)

            print(e)

            noisy_file, clean_file, noise_file, metadata = self._get_filenames(n + 1)

            sampling_frequency, noisy_audio = read(noisy_file)
            _, clean_audio = read(clean_file)
            _, noise_audio = read(noise_file)

        # noisy_audio, sampling_frequency = sf.read(noisy_file)
        # clean_audio, _ = sf.read(clean_file)
        # noise_audio, _ = sf.read(noise_file)
        num_samples = 30 * sampling_frequency  # 30 sec data
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




def save_metadata():
    import pathlib
    root = "datasets/test_set_1"
    
    # Get all .wav files in the noisy folder
    # noisy_files = glob.glob(os.path.join(root, "noisy", "*.wav"))
    # len(noisy_files)
    noisy_files = pathlib.Path(os.path.join(root, "noisy")).glob("*.wav")
    
    file_id_from_name = re.compile(r"fileid_(\d+)")
    snr_from_name = re.compile(r"snr(-?\d+)")
    target_level_from_name = re.compile(r"tl(-?\d+)")
    source_info_from_name = re.compile(r"^(.*?)_snr")
    file_metadata = {}
    for file in noisy_files:
        filename = file.name
    
        # Extract metadata using regular expressions
        file_id_match = file_id_from_name.search(filename)
        snr_match = snr_from_name.search(filename)
        target_level_match = target_level_from_name.search(filename)
        source_info_match = source_info_from_name.search(filename)
    
        # Check if all patterns are found, and if so, save the metadata
        if file_id_match and snr_match and target_level_match and source_info_match:
            file_id = file_id_match.group(1)
            snr = snr_match.group(1)
            target_level = target_level_match.group(1)
            source_info = source_info_match.group(1)
    
            # Store metadata as a dictionary
            file_metadata[file_id] = {
                "filename": filename,
                "snr": snr,
                "target_level": target_level,
                "source_info": source_info
            }
        else:
            print(f"Could not extract metadata from {filename}")
    
    import json
    # Save the metadata to a JSON file
    with open(os.path.join(root, "file_metadata.json"), "w") as json_file:
        json.dump(file_metadata, json_file, indent=4)
        
if __name__ == "__main__":
    # train_set = DNSAudio(root="../../data/MicrosoftDNS_4_ICASSP/training_set/")
    # validation_set = DNSAudio(root="../../data/MicrosoftDNS_4_ICASSP/validation_set/")
    test_set = DNSAudio(root=r"datasets/test_set_1/", normalize=True)
    len(test_set)
    ns,c,n = test_set[100]
    import matplotlib.pyplot as plt
    plt.plot(ns),plt.show()
    plt.plot(c),plt.show()
    plt.plot(n),plt.show()
    
    


        


