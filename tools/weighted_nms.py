import sys

sys.path.insert(0, '../')

import pickle
import torch
from pcdet.ops.iou3d_nms import iou3d_nms_utils
import numpy as np
import pickle
import os

def read_pkl(pkl_file):
    # Open the pickle file in binary read mode
    with open(pkl_file, 'rb') as f:
        data = pickle.load(f)
    return data

# Function for IoU calculation
def calculate_iou_rotated(boxes1, boxes2):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    boxes1 = boxes1.to(device)
    boxes2 = boxes2.to(device)
    ious = iou3d_nms_utils.boxes_iou3d_gpu(boxes1, boxes2)
    return ious

# Weighted NMS function for rotated 3D bounding boxes
def weighted_nms_rotated(boxes, scores, iou_thresh=0.5):
    sorted_indices = torch.argsort(scores, descending=True)
    selected_boxes = []
    selected_scores = []

    while len(sorted_indices) > 0:
        current_idx = sorted_indices[0]
        current_box = boxes[current_idx].unsqueeze(0)
        current_score = scores[current_idx]
        selected_boxes.append(current_box)
        selected_scores.append(current_score)

        if len(sorted_indices) == 1:
            break

        rest_indices = sorted_indices[1:]
        rest_boxes = boxes[rest_indices]
        ious = calculate_iou_rotated(current_box, rest_boxes)

        weights = torch.exp(-(ious ** 2) / iou_thresh)
        weights = weights.to(scores.device)
        scores[rest_indices] = scores[rest_indices] * weights.squeeze()
        # scores[rest_indices] = scores[rest_indices] * weights.squeeze()
        sorted_indices = rest_indices[scores[rest_indices] > 0]

    return torch.cat(selected_boxes, dim=0), torch.stack(selected_scores)

# Combine predictions from multiple models and apply NMS with class-specific IoU thresholds
def combine_and_nms_results(pred_results_list, iou_thresholds, model_weights):
    all_boxes = []
    all_scores = []
    all_labels = []

    # for pred_result in pred_results_list:
    #     boxes = torch.tensor(pred_result['boxes_lidar'], dtype=torch.float32)
    #     scores = torch.tensor(pred_result['score'], dtype=torch.float32)
    #     labels = torch.tensor(pred_result['pred_labels'], dtype=torch.int32)

    #     all_boxes.append(boxes)
    #     all_scores.append(scores)
    #     all_labels.append(labels)

    # Adjust each model's scores based on its model weight
    for pred_result, model_weight in zip(pred_results_list, model_weights):
        boxes = torch.tensor(pred_result['boxes_lidar'], dtype=torch.float32)
        scores = torch.tensor(pred_result['score'], dtype=torch.float32) * model_weight
        labels = torch.tensor(pred_result['pred_labels'], dtype=torch.int32)

        all_boxes.append(boxes)
        all_scores.append(scores)
        all_labels.append(labels)

    all_boxes = torch.cat(all_boxes, dim=0)
    all_scores = torch.cat(all_scores, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    final_boxes = []
    final_scores = []
    final_labels = []

    for label in torch.unique(all_labels):
        class_indices = all_labels == label
        class_boxes = all_boxes[class_indices]
        class_scores = all_scores[class_indices]

        # Retrieve the specific IoU threshold for this class label
        iou_thresh = iou_thresholds.get(label.item(), 0.5)  # Default to 0.5 if not specified

        # Apply weighted NMS for the current class
        selected_boxes, selected_scores = weighted_nms_rotated(class_boxes, class_scores, iou_thresh)

        final_boxes.append(selected_boxes)
        final_scores.append(selected_scores)
        final_labels.extend([label] * len(selected_scores))

    final_boxes = torch.cat(final_boxes, dim=0)
    final_scores = torch.cat(final_scores, dim=0)
    final_labels = torch.tensor(final_labels, dtype=torch.int32)

    return final_boxes, final_scores, final_labels


def main():
    pkl_file = "../dsvt_results/result.pkl"  # Path to your .pkl file
    pred_results_1 = read_pkl(pkl_file)

    pkl_file = "../pvplus_results/result.pkl"  # Path to your .pkl file
    pred_results_2 = read_pkl(pkl_file)


    # Class-specific IoU thresholds
    iou_thresholds = {
        1: 0.7,  # Vehicle
        2: 0.6,  # Pedestrian
        3: 0.55   # Cyclist
    }
    model_weights = [1.0, 0.9]

    results = []
    if not os.path.exists("../final_results"):
        os.makedirs("../final_results")
    for idx, pred_result_1 in enumerate(pred_results_1):
        pred_result_2 = pred_results_2[idx]
        if pred_result_1["frame_id"] != pred_result_2["frame_id"]:
            print("does not same of frame-id")
            break
        final_boxes, final_scores, final_labels = combine_and_nms_results([pred_result_1, pred_result_2], iou_thresholds, model_weights)
        frame_result = {
            'name': np.array(['Vehicle' if lbl == 1 else 'Pedestrian' if lbl == 2 else 'Cyclist'
                            for lbl in final_labels.numpy()]),
            'score': final_scores.cpu().numpy(),
            'boxes_lidar': final_boxes.cpu().numpy(),
            'pred_labels': final_labels.cpu().numpy(),
            'frame_id': pred_result_1["frame_id"]
        }
        print(f'frame_id  : {pred_result_1["frame_id"]}')
        results.append(frame_result)
    save_filename = "../final_results/result.pkl"
    with open(save_filename, 'wb') as f:
        pickle.dump(results, f)
if __name__ == "__main__":
    main()
