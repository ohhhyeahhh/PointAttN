import os
import open3d as o3d
import torch
import numpy as np
import torch.utils.data as data
import h5py
import math
import transforms3d
import random
from tensorpack import dataflow

class PCN_pcd(data.Dataset):
    def __init__(self, prefix="train"):
        if prefix=="train":
            self.file_path = '/data0/guodongyan/completion_dataset//ShapeNetCompletion/train'
        elif prefix=="val":
            self.file_path = '/data0/guodongyan/completion_dataset//ShapeNetCompletion/val'
        elif prefix=="test":
            self.file_path = '/data0/guodongyan/completion_dataset//ShapeNetCompletion/test'
        else:
            raise ValueError("ValueError prefix should be [train/val/test] ")

        self.prefix = prefix
        self.label_map ={'02691156': '0', '02933112': '1', '02958343': '2',
                         '03001627': '3', '03636649': '4', '04256520': '5',
                         '04379243': '6', '04530566': '7', 'all': '8'}
        
        self.label_map_inverse ={'0': '02691156', '1': '02933112', '2': '02958343',
                         '3': '03001627', '4': '03636649', '5': '04256520',
                         '6': '04379243', '7': '04530566', '8': 'all'}

        self.input_data, self.labels = self.get_data(os.path.join(self.file_path, 'partial'))

        random.shuffle(self.input_data)

        self.len = len(self.input_data)

        self.scale = 0
        self.mirror = 1
        self.rot = 0
        self.sample = 1

    def __len__(self):
        return self.len

    def read_pcd(self, path):
        pcd = o3d.io.read_point_cloud(path)
        points = np.asarray(pcd.points)
        return points

    def get_data(self, path):
        cls = os.listdir(path)
        data = []
        labels = []
        for c in cls:
            objs = os.listdir(os.path.join(path, c))
            for obj in objs:
                f_names = os.listdir(os.path.join(path, c, obj))
                obj_list = []
                for f_name in f_names:
                    data_path = os.path.join(path, c, obj, f_name)
                    obj_list.append(data_path)
                    # points = self.read_pcd(os.path.join(path, c, obj, f_name))
                data.append(obj_list)
                labels.append(self.label_map[c])


        return data, labels

    def randomsample(self, ptcloud ,n_points):
        choice = np.random.permutation(ptcloud.shape[0])
        ptcloud = ptcloud[choice[:n_points]]

        if ptcloud.shape[0] < n_points:
            zeros = np.zeros((n_points - ptcloud.shape[0], 3))
            ptcloud = np.concatenate([ptcloud, zeros])
        return ptcloud

    def upsample(self, ptcloud, n_points):
        curr = ptcloud.shape[0]
        need = n_points - curr

        if need < 0:
            return ptcloud[np.random.permutation(n_points)]

        while curr <= need:
            ptcloud = np.tile(ptcloud, (2, 1))
            # ptcloud = np.concatenate([ptcloud,np.zeros_like(ptcloud)],dim=0)
            need -= curr
            curr *= 2

        choice = np.random.permutation(need)
        ptcloud = np.concatenate((ptcloud, ptcloud[choice]))

        return ptcloud


    def get_transform(self, points):
        result = []
        rnd_value = np.random.uniform(0, 1)

        if self.mirror and self.prefix == 'train':
            trfm_mat = transforms3d.zooms.zfdir2mat(1)
            trfm_mat_x = np.dot(transforms3d.zooms.zfdir2mat(-1, [1, 0, 0]), trfm_mat)
            trfm_mat_z = np.dot(transforms3d.zooms.zfdir2mat(-1, [0, 0, 1]), trfm_mat)
            if rnd_value <= 0.25:
                trfm_mat = np.dot(trfm_mat_x, trfm_mat)
                trfm_mat = np.dot(trfm_mat_z, trfm_mat)
            elif rnd_value > 0.25 and rnd_value <= 0.5:    # lgtm [py/redundant-comparison]
                trfm_mat = np.dot(trfm_mat_x, trfm_mat)
            elif rnd_value > 0.5 and rnd_value <= 0.75:
                trfm_mat = np.dot(trfm_mat_z, trfm_mat)
        for ptcloud in points:
            ptcloud[:, :3] = np.dot(ptcloud[:, :3], trfm_mat.T)

            if self.scale:
                ptcloud = ptcloud * self.scale
            result.append(ptcloud)

        return result[0],result[1]

    def __getitem__(self, index):

        partial_path = self.input_data[index]
        n_sample = len(partial_path)
        idx = random.randint(0, n_sample-1)
        partial_path = partial_path[idx]

        partial = self.read_pcd(partial_path)

        # if self.prefix == 'train' and self.sample:
        partial = self.upsample(partial, 2048)

        gt_path = partial_path.replace('/'+partial_path.split('/')[-1],'.pcd')
        gt_path = gt_path.replace('partial','complete')


        if self.prefix == 'train':
            complete = self.read_pcd(gt_path)
            partial, complete = self.get_transform([partial, complete])
        else:
            complete = self.read_pcd(gt_path)

        complete = torch.from_numpy(complete)
        partial = torch.from_numpy(partial)
        label = partial_path.split('/')[-3]
        label = self.label_map[label]
        obj = partial_path.split('/')[-2]
        
        if self.prefix == 'test':
            return label, partial, complete, obj
        else:
            return label, partial, complete


class C3D_h5(data.Dataset):
    def __init__(self, prefix="train"):
        if prefix=="train":
            self.file_path = '/data0/guodongyan/completion_dataset/c3d/shapenet/train'
        elif prefix=="val":
            self.file_path = '/data0/guodongyan/completion_dataset/c3d/shapenet/val'
        elif prefix=="test":
            self.file_path = '/data0/guodongyan/completion_dataset/c3d/shapenet/test'
        else:
            raise ValueError("ValueError prefix should be [train/val/test] ")

        self.prefix = prefix
        self.label_map ={'02691156': '0', '02933112': '1', '02958343': '2',
                         '03001627': '3', '03636649': '4', '04256520': '5',
                         '04379243': '6', '04530566': '7', 'all': '8'}

        if prefix is not "test":
            self.input_data, self.labels = self.get_data(os.path.join(self.file_path, 'partial'))
            self.gt_data, _ = self.get_data(os.path.join(self.file_path, 'gt'))
            print(len(self.gt_data), len(self.labels))
        else:
            self.input_data, self.labels = self.get_data(os.path.join(self.file_path, 'partial'))

        print(len(self.input_data))

        self.len = len(self.input_data)

        self.scale = 1
        self.mirror = 1
        self.rot = 0
        self.sample = 1

    def __len__(self):
        return self.len

    def get_data(self, path):
        cls = os.listdir(path)
        data = []
        labels = []
        for c in cls:
            objs = os.listdir(os.path.join(path, c))
            for obj in objs:
                data.append(os.path.join(path,c,obj))
                if self.prefix == "test":
                    labels.append(obj)
                else:
                    labels.append(self.label_map[c])

        return data, labels


    def get_transform(self, points):
        result = []
        rnd_value = np.random.uniform(0, 1)
        angle = random.uniform(0,2*math.pi)
        scale = np.random.uniform(1/1.6, 1)

        trfm_mat = transforms3d.zooms.zfdir2mat(1)
        if self.mirror and self.prefix == 'train':

            trfm_mat_x = np.dot(transforms3d.zooms.zfdir2mat(-1, [1, 0, 0]), trfm_mat)
            trfm_mat_z = np.dot(transforms3d.zooms.zfdir2mat(-1, [0, 0, 1]), trfm_mat)
            if rnd_value <= 0.25:
                trfm_mat = np.dot(trfm_mat_x, trfm_mat)
                trfm_mat = np.dot(trfm_mat_z, trfm_mat)
            elif rnd_value > 0.25 and rnd_value <= 0.5:    # lgtm [py/redundant-comparison]
                trfm_mat = np.dot(trfm_mat_x, trfm_mat)
            elif rnd_value > 0.5 and rnd_value <= 0.75:
                trfm_mat = np.dot(trfm_mat_z, trfm_mat)
        if self.rot:
                trfm_mat = np.dot(transforms3d.axangles.axangle2mat([0,1,0],angle), trfm_mat)
        for ptcloud in points:
            ptcloud[:, :3] = np.dot(ptcloud[:, :3], trfm_mat.T)

            if self.scale:
                ptcloud = ptcloud * scale
            result.append(ptcloud)

        return result[0],result[1]

    def __getitem__(self, index):
        partial_path = self.input_data[index]
        with h5py.File(partial_path, 'r') as f:
            partial = np.array(f['data'])

        if self.prefix == 'train' and self.sample:
            choice = np.random.permutation((partial.shape[0]))
            partial = partial[choice[:2048]]
            if partial.shape[0] < 2048:
                zeros = np.zeros((2048-partial.shape[0],3))
                partial = np.concatenate([partial,zeros])

        if self.prefix not in ["test"]:
            complete_path = partial_path.replace('partial','gt')
            with h5py.File(complete_path, 'r') as f:
                complete = np.array(f['data'])

            partial, complete = self.get_transform([partial, complete])

            complete = torch.from_numpy(complete)
            label = (self.labels[index])
            partial = torch.from_numpy(partial)

            return label, partial, complete
        else:
            partial = torch.from_numpy(partial)
            label = (self.labels[index])
            return label, partial, partial


if __name__ == '__main__':
    dataset = C3D_h5(prefix='test')
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1,
                                             shuffle=True, num_workers=0)
    for idx, data in enumerate(dataloader, 0):
        print(data.shape)




