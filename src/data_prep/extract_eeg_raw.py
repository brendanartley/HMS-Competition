import numpy as np
import pandas as pd
import os, shutil
from tqdm import tqdm

from src.configs.cfg_wavenet import cfg

from src.utils.helpers import sample_equally_spaced

def main():

    # Vars
    sample_per_sec = 200

    # Sample EEGs w/ non-overlap
    df = pd.concat([
        pd.read_csv(os.path.join(cfg.data_dir, "processed/train_2.0sec_nooverlap.csv")),
        # pd.read_csv(os.path.join(cfg.data_dir, "processed/train_4.0sec_nooverlap.csv")),
        # pd.read_csv(os.path.join(cfg.data_dir, "processed/train_6.0sec_nooverlap.csv")),
        # pd.read_csv(os.path.join(cfg.data_dir, "processed/train_10.0sec_nooverlap.csv")),
        # pd.read_csv(os.path.join(cfg.data_dir, "processed/train_20.0sec_nooverlap.csv")),
        # pd.read_csv(os.path.join(cfg.data_dir, "processed/train_50.0sec_nooverlap.csv")),
        # pd.read_csv(os.path.join(cfg.data_dir, "processed/train_50sec_nooverlap.csv")),
    ]).drop_duplicates()
    train_dir= "train_v7"

    # Load and save 50sec EEG data
    for i, (eeg_id, eeg_label_offset_seconds, label_id) in enumerate(tqdm(zip(
        df["eeg_id"].values,
        df["eeg_label_offset_seconds"].values,
        df["label_id"].values,
    ), total=len(df))):
        
        # Load EEG
        eeg = pd.read_parquet(os.path.join(cfg.data_dir, "raw/train_eegs/", "{}.parquet".format(eeg_id)))
            
        # Get window
        start = int(eeg_label_offset_seconds*sample_per_sec)
        end = int((eeg_label_offset_seconds+50)*sample_per_sec)

        eeg_sample = eeg.iloc[start:end].values.astype(np.float32)
        assert len(eeg_sample) == 10000

        # Normalize
        for j in range(eeg_sample.shape[1]):
            x = eeg_sample[:, j]
            m = np.nanmean(x)

            # Set all 0s if all NANs
            if np.isnan(x).mean()<1: x = np.nan_to_num(x, nan=m)
            else: x[:] = 0

            eeg_sample[:, j] = x

        # Create outdir
        outdir = os.path.join(cfg.data_dir, "processed/{}/{}".format(train_dir, label_id))
        if not os.path.exists(outdir):
            os.mkdir(outdir)
        
        # Spect
        out_path = os.path.join(outdir, "eeg_ekg_raw.npy")
        np.save(out_path, eeg_sample)
            
        # # Debug run
        # if i == 0:
        #     break
    
    return


if __name__ == "__main__":
    main()