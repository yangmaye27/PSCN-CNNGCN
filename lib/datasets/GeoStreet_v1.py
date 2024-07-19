import cv2
import torch
import torch.utils.data as data
import pandas as pd
import numpy as np
from torch_geometric.data import Data
import torchvision.transforms as transforms
#
# transform1_quant = transforms.Compose([
#
#     transforms.Normalize(mean=(0.17455307, 0, 0.49139475), std=(0.02596996, 1, 0.25268278))
# ])
#
# transform1_vis = transforms.Compose([
#
#     transforms.Normalize(mean=(0.69412405, 0.70026201, 0.70355724, 0.69743059),
#                          std=(0.05905648, 0.06046197, 0.06075483, 0.06129687))
# ])


class  GEOSTREET_V1(data.Dataset):
    def __init__(self, cfg, is_train=True, transform=None):
        
        if is_train:
            self.csv_file = cfg.DATASET.TRAINSET
        else:
            self.csv_file = cfg.DATASET.TESTSET

        self.is_train = is_train
        self.data_root = cfg.DATASET.ROOT

        self.SampleNames = pd.read_csv(self.csv_file)

    def __len__(self):
        return len(self.SampleNames)

    def __getitem__(self, idx):
        if self.is_train:
            #Sample_path = './' + self.SampleNames.iloc[idx, 0]
            Sample_path ='./' + self.SampleNames.iloc[idx, 0]
        else:
            #Sample_path = './' + self.SampleNames.iloc[idx, 0]
            Sample_path ='./' +self.SampleNames.iloc[idx, 0]

        Samplenpz = np.load(Sample_path)
        Sample_In = Samplenpz['IN']
        Sample_GT256 = Samplenpz['GT_256']
        Sample_GT64JC_ALL = Samplenpz['GT_64_JC']
        Sample_GT64ST = Samplenpz['GT_64_ST']

        node_features = torch.tensor(Samplenpz['node_features'], dtype=torch.float)
        edge_index = torch.tensor(Samplenpz['edge_index'], dtype=torch.long)
        edge_attr = torch.tensor(Samplenpz['edge_attr'], dtype=torch.float)

        Sample_In = Sample_In.astype(np.float32)
        Sample_In = Sample_In / 127.5
        # 道路
        Sample_In_RN = Sample_In[:, :, 0:1]
        # 地理信息通道
        Sample_In_Quant = Sample_In[:, :, 1:4].copy()
        # hillshade四个通道
        Sample_In_Vis = Sample_In[:, :, 5:9].copy()
        # 引导节点
        Sample_In_Guid = Sample_In[:, :, 9:10]
        # 用地
        Sample_In_ND = Sample_In[:, :, 4:5]


        Sample_In_Quant_temp = torch.from_numpy(Sample_In_Quant.transpose((2, 0, 1)))
        Sample_In_Vis_temp = torch.from_numpy(Sample_In_Vis.transpose((2, 0, 1)))

        Slopmap = (Sample_In_Quant_temp[1, :, :].clone() + 1) / 2
        Slopmap = torch.Tensor(cv2.blur(Slopmap.numpy(), (5, 5)))
        Slopmap[Slopmap > 0.15] = 1
        Slopmap[Slopmap < 1] = 0


        Sample_In_RN = torch.from_numpy(Sample_In_RN.transpose([2, 0, 1]))
        Sample_In_Guid = torch.from_numpy(Sample_In_Guid.transpose([2, 0, 1]))
        Sample_In_ND = torch.from_numpy(Sample_In_ND.transpose([2, 0, 1]))

        # Sample_GT256 = Sample_GT256.astype(np.float32)
        # Sample_GT256 = Sample_GT256 / 127.5
        # Sample_GT256 = Sample_GT256.transpose([2, 0, 1])
        #
        # Sample_GT64JC_ALL = Sample_GT64JC_ALL.astype(np.float32)
        # Sample_GT64JC_ALL = Sample_GT64JC_ALL / 127.5
        # Sample_GT64JC_ALL = Sample_GT64JC_ALL.transpose([2, 0, 1])
        # Sample_GT64JC_ALL = (torch.Tensor(Sample_GT64JC_ALL) + 1) / 2 * Slopmap.clone() * 2 - 1
        #
        # Sample_GT64ST = Sample_GT64ST.astype(np.float32)
        # Sample_GT64ST = Sample_GT64ST / 127.5
        # Sample_GT64ST = Sample_GT64ST.transpose([2, 0, 1])

        Sample_GT256 = torch.from_numpy(Sample_GT256.astype(np.float32) / 127.5).permute(2, 0, 1)
        Sample_GT64JC_ALL = torch.from_numpy(Sample_GT64JC_ALL.astype(np.float32) / 127.5).permute(2, 0, 1)
        Sample_GT64JC_ALL = (Sample_GT64JC_ALL + 1) / 2 * Slopmap.clone() * 2 - 1
        Sample_GT64ST = torch.from_numpy(Sample_GT64ST.astype(np.float32) / 127.5).permute(2, 0, 1)

        # print(Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr))
        return Sample_In_RN, Sample_In_Quant_temp, Sample_In_Vis_temp, Sample_In_Guid, Sample_In_ND,\
               Sample_GT256, Sample_GT64JC_ALL, Sample_GT64ST, Slopmap,Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr)


if __name__ == '__main__':
    pass
