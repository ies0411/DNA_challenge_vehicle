import _init_path

import argparse
import glob
from pathlib import Path
import numpy as np
import torch
import os
from tqdm import tqdm
from easydict import EasyDict

from pcdet.config import cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils
# from pcdet.utils.nms_3d import nms
import pickle

from pcdet.ops.iou3d_nms import iou3d_nms_utils

def calculate_iou_rotated(boxes1, boxes2):
    """
    Calculate IoU for rotated 3D boxes using OpenPCDet's IoU function.

    Args:
        boxes1 (torch.Tensor): First set of boxes (N, 7) in [x, y, z, dx, dy, dz, heading]
        boxes2 (torch.Tensor): Second set of boxes (M, 7) in [x, y, z, dx, dy, dz, heading]

    Returns:
        ious (torch.Tensor): IoU values between each pair of boxes (N, M)
    """
    # Calculate IoU between two sets of boxes
    ious = iou3d_nms_utils.boxes_iou3d_gpu(boxes1, boxes2)
    return ious

def weighted_nms_rotated(boxes, scores, iou_thresh=0.5):
    """
    Apply weighted NMS for rotated 3D bounding boxes using IoU-based suppression.

    Args:
        boxes (torch.Tensor): 3D boxes (N, 7) in [x, y, z, dx, dy, dz, heading]
        scores (torch.Tensor): Confidence scores for each box (N,)
        iou_thresh (float): IoU threshold for suppression.

    Returns:
        selected_boxes (list): List of final selected boxes after weighted NMS.
        selected_scores (list): List of final selected scores.
    """
    # Sort boxes by scores in descending order
    sorted_indices = torch.argsort(scores, descending=True)
    selected_boxes = []
    selected_scores = []

    while len(sorted_indices) > 0:
        # Take the box with the highest score
        current_idx = sorted_indices[0]
        current_box = boxes[current_idx].unsqueeze(0)
        current_score = scores[current_idx]
        selected_boxes.append(current_box)
        selected_scores.append(current_score)

        if len(sorted_indices) == 1:
            break

        # Calculate IoU with the remaining boxes
        rest_indices = sorted_indices[1:]
        rest_boxes = boxes[rest_indices]
        ious = calculate_iou_rotated(current_box, rest_boxes)

        # Apply weight adjustment based on IoU
        weights = torch.exp(-(ious ** 2) / iou_thresh)  # Apply Gaussian-like weighting
        scores[rest_indices] = scores[rest_indices] * weights.squeeze()

        # Filter boxes with scores above a threshold
        sorted_indices = rest_indices[scores[rest_indices] > 0]

    return torch.cat(selected_boxes, dim=0), torch.stack(selected_scores)

# Example usage
# boxes = torch.tensor([[0, 0, 0, 1, 1, 1, 0], [0.1, 0.1, 0.1, 1, 1, 1, 0.1], [1, 1, 1, 1, 1, 1, 0.5]])  # Example boxes
# scores = torch.tensor([0.9, 0.8, 0.7])  # Example scores
# iou_thresh = 0.5

# selected_boxes, selected_scores = weighted_nms_rotated(boxes, scores, iou_thresh)

# print("Selected boxes:", selected_boxes)
# print("Selected scores:", selected_scores)


def get_filename_without_extension(file_path):
    """파일 전체 경로에서 확장자를 제외한 파일명을 추출합니다."""
    filename_with_extension = os.path.basename(file_path)
    filename_without_extension = os.path.splitext(filename_with_extension)[0]
    return filename_without_extension

class DemoDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None, ext='.npy'):
        """
        Args:
            dataset_cfg (dict): 데이터셋 설정.
            class_names (list): 클래스 이름들.
            training (bool): 학습 여부.
            root_path (Path or str): 데이터셋 루트 경로.
            logger (Logger): 로거.
            ext (str): 파일 확장자.
        """
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.root_path = root_path
        self.ext = ext

        test_frame_ids_filename = os.path.join(root_path, "ImageSets/test.txt")
        with open(test_frame_ids_filename, 'r') as f:
            test_frame_ids = [line.strip() for line in f.readlines()]

        self.sample_file_list = sorted(
            [os.path.join(root_path, f"points/{frame_id}{ext}") for frame_id in test_frame_ids]
        )

    def __len__(self):
        return len(self.sample_file_list)

    def __getitem__(self, index):
        if self.ext == '.bin':
            points = np.fromfile(self.sample_file_list[index], dtype=np.float32).reshape(-1, 4)
        elif self.ext == '.npy':
            points = np.load(self.sample_file_list[index])
        else:
            raise NotImplementedError(f"File extension {self.ext} is not supported")

        frame_id = get_filename_without_extension(self.sample_file_list[index])
        input_dict = {'points': points, 'frame_id': frame_id}

        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict

def parse_config():
    parser = argparse.ArgumentParser(description='Argument parser for demo')
    parser.add_argument('--cfg_file_1', type=str, default='./cfgs/custom_models/dsvt_deep.yaml', help='Specify the config for demo')
    parser.add_argument('--cfg_file_2', type=str, default="./cfgs/custom_models/pvrcnn_plus_res.yaml", help='Specify the config for demo')
    parser.add_argument('--data_path', type=str, default="/mnt/nas-1/eslim/Data/dna/vehicle/custom/", help='Specify the custom_av directory')
    parser.add_argument('--ckpt_1', type=str, default="/mnt/nas-1/eslim/workspace/dna_vehicle/dsvt_deep_downsample_full/ckpt/checkpoint_epoch_6.pth", help='Specify the pretrained model')
    parser.add_argument('--ckpt_2', type=str, default="/mnt/nas-1/eslim/workspace/dna_vehicle/pv_res_down_bal_full/ckpt/checkpoint_epoch_5.pth", help='Specify the pretrained model')
    parser.add_argument('--ext', type=str, default='.npy', help='Specify the extension of your point cloud data file')
    # parser.add_argument('--work_dir', type=str)

    args = parser.parse_args()
    cfg_1 = EasyDict()
    cfg_2 = EasyDict()

    cfg_from_yaml_file(args.cfg_file_1, cfg_1)
    cfg_from_yaml_file(args.cfg_file_2, cfg_2)
    return args, cfg_1, cfg_2

def main():
    args, cfg_1, cfg_2 = parse_config()
    logger = common_utils.create_logger()

    demo_dataset_1 = DemoDataset(
        dataset_cfg=cfg_1.DATA_CONFIG, class_names=cfg_1.CLASS_NAMES, training=False,
        root_path=Path(args.data_path), ext=args.ext, logger=logger
    )
    demo_dataset_2 = DemoDataset(
        dataset_cfg=cfg_2.DATA_CONFIG, class_names=cfg_2.CLASS_NAMES, training=False,
        root_path=Path(args.data_path), ext=args.ext, logger=logger
    )
    logger.info(f'Total number of samples: \t{len(demo_dataset_1)}')
    logger.info(f'Total number of samples: \t{len(demo_dataset_2)}')

    model_1 = build_network(model_cfg=cfg_1.MODEL, num_class=len(cfg_1.CLASS_NAMES), dataset=demo_dataset_1)
    model_1.load_params_from_file(filename=args.ckpt_1, logger=logger, to_cpu=True)
    model_1.cuda()
    model_1.eval()

    model_2 = build_network(model_cfg=cfg_2.MODEL, num_class=len(cfg_2.CLASS_NAMES), dataset=demo_dataset_2)
    model_2.load_params_from_file(filename=args.ckpt_2, logger=logger, to_cpu=True)
    model_2.cuda()
    model_2.eval()
    # if not os.path.exists(args.work_dir):
        # os.makedirs(args.work_dir)

    # save_filename = "%s/result.pkl"%(args.work_dir)

    # det_annos = list()
    file_path_prefix = "../datasets/custom/labels"
    threshold_score = 0.78
    with torch.no_grad():
        for idx, data_dict_1 in enumerate(tqdm(demo_dataset_1, desc="Processing dataset")):
            data_dict_1 = demo_dataset_1.collate_batch([data_dict_1])
            load_data_to_gpu(data_dict_1)
            pred_dicts_1, _ = model_1.forward(data_dict_1)
            pred_dict_1 = pred_dicts_1[0]
            frame_id_1 = data_dict_1['frame_id'][0]

            pred_dict_1 = {k: v.cpu().numpy() for k, v in pred_dict_1.items()}

            num_obj_1 = len(pred_dict_1['pred_labels'])

            if num_obj_1 == 0:
                print("At least one object should be detected.")
                assert num_obj_1 != 0, "No objects detected. The program requires at least one object to be detected."


            data_dict_2 = demo_dataset_2[idx]
            data_dict_2 = demo_dataset_2.collate_batch([data_dict_2])
            load_data_to_gpu(data_dict_2)
            pred_dicts_2, _ = model_2.forward(data_dict_2)
            pred_dict_2 = pred_dicts_2[0]
            frame_id_2 = data_dict_2['frame_id'][0]
            pred_dict_2 = {k: v.cpu().numpy() for k, v in pred_dict_2.items()}

            num_obj_2 = len(pred_dict_2['pred_labels'])

            if num_obj_2 == 0:
                print("At least one object should be detected.")
                assert num_obj_2 != 0, "No objects detected. The program requires at least one object to be detected."


            assert frame_id_1 == frame_id_2 ,  "frame id is differenct"
            # print(f'processing : {frame_id_1}')
            file_path = os.path.join(file_path_prefix,str(frame_id_1)+".txt")
            # if str(frame_id_1) == str(10008994):
            #     print("check")
            #     print(num_obj_1)
            #     print(num_obj_2)
            with open(file_path, "w") as file:
                for obj_idx in range(num_obj_1):
                    if pred_dict_1["pred_scores"][obj_idx] < threshold_score or pred_dict_1['pred_labels'][obj_idx] == 1:
                        continue
                    file.write(" ".join(map(str, pred_dict_1["pred_boxes"][obj_idx])) + " ")

                    class_id = pred_dict_1['pred_labels'][obj_idx] - 1
                    class_name = cfg_1.CLASS_NAMES[class_id]
                    file.write(class_name+"\n")

                for obj_idx in range(num_obj_2):
                    if pred_dict_2["pred_scores"][obj_idx] < threshold_score or pred_dict_2['pred_labels'][obj_idx] != 1:
                        continue
                    file.write(" ".join(map(str, pred_dict_2["pred_boxes"][obj_idx])) + " ")

                    class_id = pred_dict_2['pred_labels'][obj_idx] - 1
                    class_name = cfg_2.CLASS_NAMES[class_id]
                    file.write(class_name+"\n")
            # exit()
            print(f'finish : {frame_id_1}')
if __name__ == '__main__':
    main()


# python test.py --cfg_file ./cfgs/custom_models/dsvt_deep.yaml --ckpt /mnt/nas-1/eslim/workspace/dna_edge/dsvt_deep_2/ckpt/checkpoint_epoch_24.pth
# --data_path /mnt/nas-1/eslim/Data/dna/edge/custom/  --work_dir /mnt/nas-1/eslim/workspace/dna_edge/results/test


# empty num : 10002747
# empty num : 10002778
# empty num : 10002802
# empty num : 10002804
# empty num : 10002803
# empty num : 10002805
# empty num : 10002806
# empty num : 10002807
# empty num : 10002808
# empty num : 10002809
# empty num : 10002811
# empty num : 10002812
# empty num : 10005032
# empty num : 10005037
# empty num : 10005041
# empty num : 10005129
# empty num : 10005177
# empty num : 10005179
# empty num : 10005194
# empty num : 10005195
# empty num : 10005196
# empty num : 10005197
# empty num : 10005203
# empty num : 10005852
# empty num : 10005866
# empty num : 10005865
# empty num : 10005869
# empty num : 10005870
# empty num : 10005872
# empty num : 10005874
# empty num : 10005908
# empty num : 10005919
# empty num : 10005924
# empty num : 10005929
# empty num : 10005930
# empty num : 10005934
# empty num : 10005935
# empty num : 10005968
# empty num : 10005970
# empty num : 10005969
# empty num : 10005971
# empty num : 10005974
# empty num : 10005976
# empty num : 10005978
# empty num : 10005980
# empty num : 10006000
# empty num : 10006008
# empty num : 10006007
# empty num : 10006012
# empty num : 10006013
# empty num : 10006014
# empty num : 10006016
# empty num : 10006015
# empty num : 10006020
# empty num : 10006019
# empty num : 10006022
# empty num : 10006021
# empty num : 10006027
# empty num : 10006028
# empty num : 10006029
# empty num : 10006030
# empty num : 10007416
# empty num : 10007421
# empty num : 10007422
# empty num : 10007423
# empty num : 10008994
# empty num : 10008995