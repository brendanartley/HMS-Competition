import numpy as np
import pandas as pd
import os
from tqdm import tqdm

from src.configs.cfg_wavenet import cfg

from src.utils.helpers import sample_equally_spaced

def main():

    # Vars
    cur_spec_id = -1
    sample_per_sec = 200

    # Sample EEGs w/ non-overlap
    df = pd.concat([
        pd.read_csv(os.path.join(cfg.data_dir, "processed/train_2.0sec_nooverlap.csv")),
        # pd.read_csv(os.path.join(cfg.data_dir, "processed/train_4.0sec_nooverlap.csv")),
        # pd.read_csv(os.path.join(cfg.data_dir, "processed/train_6.0sec_nooverlap.csv")),
        # pd.read_csv(os.path.join(cfg.data_dir, "processed/train_10.0sec_nooverlap.csv")),
        # pd.read_csv(os.path.join(cfg.data_dir, "processed/train_20.0sec_nooverlap.csv")),
        # pd.read_csv(os.path.join(cfg.data_dir, "processed/train_50.0sec_nooverlap.csv")),
    ]).drop_duplicates()
    train_dir= "train_v7"

    # Load and save 50sec EEG data
    for i, (spec_id, spec_sub_id, spec_label_offset_seconds, label_id) in enumerate(tqdm(zip(
        df["spectrogram_id"].values,
        df["spectrogram_sub_id"].values,
        df["spectrogram_label_offset_seconds"].values,
        df["label_id"].values,
    ), total=len(df))):
        
        # ------------ Loading Spectrograms ------------ 
        # Update SPECT data if needed
        if spec_id != cur_spec_id:
            cur_spec_id = spec_id
            spec_full = pd.read_parquet(os.path.join(cfg.data_dir, "raw/train_spectrograms/", "{}.parquet".format(spec_id)))

        # Get window
        start = int(spec_label_offset_seconds)
        if start%2==0: start += 1
        end = start + 598
        spec = spec_full[(spec_full.time>=start)&(spec_full.time<=end)].copy()

        # Split into 4 brain segments
        spec = np.stack([
            spec.filter(regex='^LL', axis=1),
            spec.filter(regex='^RL', axis=1),
            spec.filter(regex='^RP', axis=1),
            spec.filter(regex='^LP', axis=1)
        ], axis=0)
        assert spec.shape == (4, 300, 100)

        # Create outdir
        outdir = os.path.join(cfg.data_dir, "processed/{}/{}".format(train_dir, label_id))
        if not os.path.exists(outdir):
            os.mkdir(outdir)

        # Spect
        out_path = os.path.join(outdir, "spect.npy")
        np.save(out_path, spec)
        
        # # Debug run
        # if i == 0:
        #     break
    
    return


if __name__ == "__main__":
    main()