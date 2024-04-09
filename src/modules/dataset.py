import os
import pickle
from tqdm import tqdm

from scipy.signal import butter, lfilter, sosfiltfilt

import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, cfg, mode="train"):
        self.cfg = cfg
        self.mode = mode

        if mode == "train":
            self.aug1d = cfg.train_aug_1d
            self.aug2d_comps = cfg.train_aug_2d_comps

        else:
            self.aug1d = cfg.val_aug_1d
            self.aug2d_comps = cfg.val_aug_2d_comps

        self.vote2label = {'gpd_vote': 0, 'grda_vote': 1, 'lpd_vote': 2, 'lrda_vote': 3, 'other_vote': 4, 'seizure_vote': 5}
        self.label2vote = {0: 'gpd_vote', 1: 'grda_vote', 2: 'lpd_vote', 3: 'lrda_vote', 4: 'other_vote', 5: 'seizure_vote'}
        self.eeg_cols = [
            'Fp1', 'F3', 'C3', 'P3', 'F7', 
            'T3', 'T5', 'O1', 'Fz', 'Cz', 
            'Pz','Fp2', 'F4', 'C4', 'P4',
            'F8', 'T4', 'T6', 'O2', 'EKG',
            ]
        self.feat2idx = {x:i for i, x in enumerate(self.eeg_cols)}
        self.idx2feat = {i:x for i, x in enumerate(self.eeg_cols)}

        # ----- Brain-Flip Augmentation Mapping -----
        self.node_leftright_map= {
            'Fp1': 'Fp2', 'Fp2': 'Fp1', 'O1': 'O2', 'O2': 'O1', 'F3': 'F4', 'F4': 'F3', 'C3': 'C4', 
            'C4': 'C3', 'P3': 'P4', 'P4': 'P3', 'T3': 'T4', 'T4': 'T3', 'T5': 'T6', 'T6': 'T5', 
            'Fz': 'Fz', 'Cz': 'Cz', 'Pz': 'Pz', 'F8': 'F7', 'F7': 'F8',
            }

        self.node_frontback_map= {
            'Fp1': 'O1', 'Fp2': 'O2', 'O1': 'Fp1', 'O2': 'Fp2', 'F3': 'P3', 'F4': 'P4', 'C3': 'C3', 
            'C4': 'C4', 'P3': 'F3', 'P4': 'F4', 'T3': 'T3', 'T4': 'T4', 'T5': 'F7', 'T6': 'F8', 
            'Fz': 'Pz', 'Cz': 'Cz', 'Pz': 'Fz', 'F8': 'T6', 'F7': 'T5',
            }

        self.diffs_dict = {
            "Bipolar 2x": [('O1', 'T3'), ('T3', 'Fp1'), ('O1', 'C3'), ('C3', 'Fp1'), ('O2', 'T4'), ('T4', 'Fp2'), ('O2','C4'), ('C4', 'Fp2')],
            "Bipolar 2xA": [('O1', 'T5'), ('T5', 'T3'), ('T3', 'F7'), ('F7', 'Fp1'), ('O2', 'T6'), ('T6', 'T4'), ('T4','F8'), ('F8', 'Fp2')],
            "Bipolar 2xB": [('O1', 'P3'), ('P3', 'C3'), ('C3', 'F3'), ('F3', 'Fp1'), ('O2', 'P4'), ('P4', 'C4'), ('C4','F4'), ('F4', 'Fp2')],
            "Transverse": [('Fz', 'F7'), ('Fz', 'F8'), ('Cz', 'T3'), ('Cz', 'T4'), ('Pz', 'T5'), ('Pz', 'T6'), ('P3', 'P4'), ('F3', 'F4')],
        }
        # Flatten list
        self.wavenet_diffpairs= [x for xs in [self.diffs_dict[x] for x in self.cfg.diffs] for x in xs]

        # Multiplier to make edges noisy
        self.inverse_gauss= 1- np.exp(-(np.arange(self.cfg.seq_len) - self.cfg.seq_len / 2) ** 2 / (2 * 3000 ** 2))

        self.oof_dict= self.load_oof_relabel()
        self.ids, self.label_dict= self.load_records()

        if self.cfg.pretrain: 
            self.data_version= "v6"
        else:  
            self.data_version= "v7"

    def load_oof_relabel(self,):
        with open('./data/oof_noisy_student.pkl', 'rb') as handle:
            oof_pred_dict = pickle.load(handle)
        oof_pred_dict = {str(k):v for k,v in oof_pred_dict.items()}
        return oof_pred_dict

    def load_records(self,):
        print("Loading records..")

        # Load data
        vote_cols = sorted(self.vote2label.keys())

        # Pretraining run
        if self.cfg.pretrain:
            csv_path= "processed/pretrain_50sec_nooverlap.csv"
        else:
            csv_path= "processed/train_50sec_nooverlap.csv"

        # Load metadata
        if self.mode == "train":
            csv_path= "processed/{}".format(self.cfg.metadata)
            df = pd.read_csv(os.path.join(self.cfg.data_dir, csv_path))
        else:
            csv_path= "processed/train_50sec_nooverlap.csv"
            df = pd.read_csv(os.path.join(self.cfg.data_dir, csv_path))
            df = df.drop_duplicates(["eeg_id", "eeg_sub_id"])
        
        # Select fold
        if self.mode == "train":
            if self.cfg.train_all_data:
                pass
            else:
                condition = df["fold"] != self.cfg.val_fold
                df = df[condition]
        else:
            if self.cfg.train_all_data:
                return [], {}
            else:
                condition = df["fold"] == self.cfg.val_fold
                df = df[condition]

        # Get labels
        labels = df[vote_cols].values
        labels = torch.from_numpy(labels).double()
        labels = labels / labels.sum(dim=1, keepdim=True) # Normalize vote ratios
        label_dict = {}
        for label_id, label in zip(df["label_id"].values, labels):
            label_dict[label_id] = label

        if self.mode == "train":
            df = df.sort_values("eeg_sub_id").groupby(["eeg_id", "spectrogram_id", "patient_id"]) \
                        .apply(lambda x: x[["label_id", "total_votes"]].values) \
                        .reset_index(name="label_id")

        else:
            df = df.sort_values("eeg_sub_id").groupby(["eeg_id", "spectrogram_id", "patient_id"]) \
                        .apply(lambda x: x[["label_id", "total_votes"]].values) \
                        .reset_index(name="label_id")

        ids = df.values
        return ids, label_dict
    
    def comp_spectrogram_scale(self, seq):
        seq = np.clip(seq, np.exp(-7), np.exp(10))
        seq = np.log(seq)
        return seq
    
    def add_difference_slides(self, seq):
        diff_arr = []
        for i in range(seq.shape[2]):
            for j in range(i+1, seq.shape[2]):
                diff_arr.append(seq[:, :, i] - seq[:, :, j])
        diff_arr = np.stack(diff_arr, axis=-1)
        seq = np.concatenate([seq, diff_arr], axis=-1)
        return seq

    def butter_lowpass_filter(self, data, cutoff_freq=20, sampling_rate=200, order=4):
        nyquist = 0.5 * sampling_rate
        normal_cutoff = cutoff_freq / nyquist
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        filtered_data = lfilter(b, a, data, axis=1)
        return filtered_data

    def get_noisy_student_labels(self, label_id, label):
        if label_id in self.oof_dict:
            return self.oof_dict[label_id].float()
        else:
            return label

    def flip_node_mapper(self, node, lr_flip, fb_flip):
        if lr_flip:
            node= self.node_leftright_map[node]
        if fb_flip:
            node= self.node_frontback_map[node]
        return node
    
    def __getitem__(self, idx):
        
        eeg_id, spect_id, patient_id, label_ids = self.ids[idx]
        return_dict = {}

        # Get label
        if self.mode == "train":
            random_idx = np.random.randint(0, len(label_ids))
            label_id, total_votes = label_ids[random_idx]

            # More weight to labels w/ higher number of votes
            return_dict["total_votes"] = torch.tensor([np.log(total_votes+1)/np.log(3)]*6).float()

        else:
            label_id, total_votes = label_ids[0] # force same row during validation
            return_dict["total_votes"] = torch.tensor([1.0]*6).float()

        label_id= int(label_id)
        label = self.label_dict[label_id].float()

        # Noisy Student Labels
        if self.mode == "train":
            if np.random.random() < self.cfg.noisy_student_prob:
                label = self.get_noisy_student_labels(str(label_id), label).float()

        return_dict["label"] = label
        return_dict["label_id"] = torch.tensor(label_id).long()
        return_dict["patient_id"] = torch.tensor(patient_id).long()

        # --- Brain-Flip Augs ---
        if self.mode == "train":
            head_scale = np.random.uniform(1-self.cfg.head_scale_range/2, 1+self.cfg.head_scale_range/2)
            lr_flip= np.random.random() < self.cfg.fb_flip_prob
            fb_flip= np.random.random() < self.cfg.fb_flip_prob
        else:
            head_scale= 1.0
            lr_flip = self.cfg.val_lr_flip
            fb_flip = self.cfg.val_fb_flip

        # --- Comp Spectrograms ---
        if self.cfg.encoder_model_type == "milAllbfg":
            fpath = os.path.join(self.cfg.data_dir, "processed/train_{}/{}/spect.npy".format(self.data_version, label_id))
            spect = np.load(fpath).transpose(1,2,0) # (seq_len, n_features, n_nodes) - (300, 100, 4)
            assert spect.shape == (300, 100, 4)

            # Flip aug
            # note: fb_flip has no effect here
            if lr_flip:
                spect[:, :, [1,0,3,2]]

            # Diffs + Scale
            spect = self.add_difference_slides(spect)
            spect = self.comp_spectrogram_scale(spect)
            
            # NP -> Torch
            if self.aug2d_comps:
                spect = self.aug2d_comps(image=spect)["image"]
            spect = spect.permute(2,0,1)
                        
            # Output dict
            if self.cfg.encoder_model_type == "milAllbfg":
                return_dict["input_comp_spectrogram"] = spect
            else:
                return_dict["input_spectrogram"] = spect
        
        # ------ WaveNet + EEG (on the fly) ------
        if self.cfg.encoder_model_type == "milAllbfg":
            fpath = os.path.join(self.cfg.data_dir, "processed/train_{}/{}/eeg_ekg_raw.npy".format(self.data_version, label_id))
            eeg = np.load(fpath).transpose(1,0)
            
            # Standardize / FillNA
            eeg = np.clip(eeg, -1024, 1024)
            eeg = np.nan_to_num(eeg, nan=0) / 32.0

            # Butter Low-Pass Filter
            eeg = self.butter_lowpass_filter(eeg, cutoff_freq=self.cfg.butter_cutoff_freq, order=self.cfg.butter_order)
            eeg = eeg.astype(np.float32)

            # AUG: Head size scaling
            if self.mode == "train":
                head_scale = np.random.uniform(1-self.cfg.head_scale_range/2, 1+self.cfg.head_scale_range/2)
            else:
                head_scale= 1.0

            # Add diffs
            res = np.zeros((len(self.wavenet_diffpairs), 10_000), dtype=np.float32)
            for i, nodes in enumerate(self.wavenet_diffpairs):
                f1, f2= nodes
                f1= self.flip_node_mapper(f1, lr_flip, fb_flip)
                f2= self.flip_node_mapper(f2, lr_flip, fb_flip)

                res[i, :] = (eeg[self.feat2idx[f1], :] - eeg[self.feat2idx[f2], :]) * head_scale

            # AUG: Noise (more noise on edges)
            if self.mode == "train":
                res += np.random.normal(0, self.cfg.gauss_noise_sigma/3, res.shape) + \
                        np.random.normal(0, self.cfg.gauss_noise_sigma, res.shape) * self.inverse_gauss
        
            # NP -> Torch
            if self.aug1d:
                res = self.aug1d(image=res)["image"]
            return_dict["input_eeg_raw"] = res

        return return_dict
    
    def __len__(self,):
        return len(self.ids)
    
if __name__ == "__main__":
    from tqdm import tqdm
    import pickle
    from src.configs.cfg_1 import cfg

    ds = CustomDataset(cfg=cfg, mode="train")
    ds = CustomDataset(cfg=cfg, mode="val")
    z= ds[0]
    for k, v in z.items():
        try: print(k, v.shape)
        except: print(k, type(v))

    # # ---- Save Data to check it matches Kaggle Env exactly ----
    # for i in tqdm(range(len(ds))):
    #     z= ds[i]
    #     z = {k:z[k] for k in ["input_comp_spectrogram", "input_eeg_raw"]}
    #     with open(f'./data/local_outputs/{i}.pickle', 'wb') as handle:
    #         pickle.dump(z, handle, protocol=pickle.HIGHEST_PROTOCOL)
    #         print("SAVE DS SAMPLE FOR TESTING..")