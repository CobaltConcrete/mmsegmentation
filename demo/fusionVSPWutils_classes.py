import os
import csv
import time
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Example Mappings
CITYSCAPES_TO_VSPW_MAPPING = {
    0: 19, 1: 20, 2: [22, 23], 3: 1, 4: 8, 5: 13, 6: 59, 7: 41, 8: [16, 67, 68, 69],
    10: 29, 11: 61, 13: 49, 14: 51, 15: 50, 17: 53, 18: 52
}

VSPW_TO_CITYSCAPES_MAPPING = {
    19: 0, 20: 1, 22: 2, 23: 2, 1: 3, 8: 4, 13: 5, 59: 6, 41: 7,
    16: 8, 67: 8, 68: 8, 69: 8, 29: 10, 61: 11, 49: 13,
    51: 14, 50: 15, 53: 17, 52: 18
}

CITYSCAPES_CLASS_NAMES = {
    0: "road", 1: "sidewalk", 2: "building", 3: "wall", 4: "fence", 5: "pole",
    6: "traffic light", 7: "traffic sign", 8: "vegetation", 10: "sky",
    11: "person", 13: "car", 14: "truck", 15: "bus", 17: "motorcycle", 18: "bicycle"
}

VALID_CLASS_IDS = [11, 13, 14, 15, 17]

def initialise_metrics_classes():
    metrics = ["mIoU", "Accuracy", "Precision", "Recall", "F1 Score", "Count"]
    return {
        f"{metric}_{CITYSCAPES_CLASS_NAMES[class_id]}": 0.0
        for class_id in VALID_CLASS_IDS
        for metric in metrics
    }


def calculate_average_metrics_classes(class_metrics, predicted_mask, gt_mask, model_to_gt_mapping, gt_to_model_mapping):
    IGNORE_LABEL = 99

    for cityscapes_id in VALID_CLASS_IDS:
        vspw_ids = CITYSCAPES_TO_VSPW_MAPPING[cityscapes_id]
        class_name = CITYSCAPES_CLASS_NAMES[cityscapes_id]

        if not isinstance(vspw_ids, list):
            vspw_ids = [vspw_ids]

        class_mask_gt = np.isin(gt_mask, vspw_ids)
        class_mask_pred = (predicted_mask == cityscapes_id)

        print(f"\nEvaluating class: {cityscapes_id} ({class_name})")

        if np.sum(class_mask_gt) == 0:
            print("  Skipping: class not in GT")
            continue

        intersection = np.logical_and(class_mask_gt, class_mask_pred).sum()
        union = np.logical_or(class_mask_gt, class_mask_pred).sum()
        iou = intersection / union if union > 0 else 0.0

        gt_binary = class_mask_gt.astype(int).flatten()
        pred_binary = class_mask_pred.astype(int).flatten()

        if np.sum(gt_binary) != 0 and np.sum(pred_binary) != 0:
            acc = accuracy_score(gt_binary, pred_binary) if len(gt_binary) > 0 else 0.0
            prec = precision_score(gt_binary, pred_binary, average='binary', zero_division=0)
            rec = recall_score(gt_binary, pred_binary, average='binary', zero_division=0)
            f1 = f1_score(gt_binary, pred_binary, average='binary', zero_division=0)

            print(f"  GT positives: {np.sum(gt_binary)}, Pred positives: {np.sum(pred_binary)}")
            print(f"  Metrics - IoU: {iou:.4f}, Acc: {acc:.4f}, Prec: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")

            # Obtain class counts
            prev_count = class_metrics[f"Count_{class_name}"]
            next_count = prev_count + 1
            # Incremental averaging for each metric
            class_metrics[f"mIoU_{class_name}"] = (class_metrics[f"mIoU_{class_name}"] * prev_count + iou) / next_count
            class_metrics[f"Accuracy_{class_name}"] = (class_metrics[f"Accuracy_{class_name}"] * prev_count + acc) / next_count
            class_metrics[f"Precision_{class_name}"] = (class_metrics[f"Precision_{class_name}"] * prev_count + prec) / next_count
            class_metrics[f"Recall_{class_name}"] = (class_metrics[f"Recall_{class_name}"] * prev_count + rec) / next_count
            class_metrics[f"F1 Score_{class_name}"] = (class_metrics[f"F1 Score_{class_name}"] * prev_count + f1) / next_count
            # Update count
            class_metrics[f"Count_{class_name}"] = next_count

    return class_metrics


def calculate_metrics_classes_old(predicted_mask, gt_mask, model_to_gt_mapping, gt_to_model_mapping):
    IGNORE_LABEL = 99
    augmented_gt_mask = np.full_like(gt_mask, IGNORE_LABEL)

    for gt_class, model_class in gt_to_model_mapping.items():
        augmented_gt_mask[gt_mask == gt_class] = model_class

    valid_pixels = (augmented_gt_mask != IGNORE_LABEL)
    filtered_pred_mask = predicted_mask[valid_pixels]
    filtered_gt_mask = augmented_gt_mask[valid_pixels]

    class_metrics = {}

    for cityscapes_id in VALID_CLASS_IDS:
        vspw_ids = CITYSCAPES_TO_VSPW_MAPPING[cityscapes_id]
        class_name = CITYSCAPES_CLASS_NAMES[cityscapes_id]

        if not isinstance(vspw_ids, list):
            vspw_ids = [vspw_ids]

        class_mask_gt = np.isin(gt_mask, vspw_ids)
        class_mask_pred = (predicted_mask == cityscapes_id)

        print(f"\nEvaluating class: {cityscapes_id} ({class_name})")

        if np.sum(class_mask_gt | class_mask_pred) == 0:
            print("  Skipping: no relevant pixels in prediction or GT")
            metrics = {f"mIoU_{class_name}": 0.0, f"Accuracy_{class_name}": 0.0,
                       f"Precision_{class_name}": 0.0, f"Recall_{class_name}": 0.0,
                       f"F1 Score_{class_name}": 0.0}
        else:
            intersection = np.logical_and(class_mask_gt, class_mask_pred).sum()
            union = np.logical_or(class_mask_gt, class_mask_pred).sum()
            iou = intersection / union if union > 0 else 0.0

            gt_binary = class_mask_gt.astype(int).flatten()
            pred_binary = class_mask_pred.astype(int).flatten()

            if np.sum(gt_binary) != 0 and np.sum(pred_binary) != 0:

                acc = accuracy_score(gt_binary, pred_binary)
                prec = precision_score(gt_binary, pred_binary, average='binary', zero_division=0)
                rec = recall_score(gt_binary, pred_binary, average='binary', zero_division=0)
                f1 = f1_score(gt_binary, pred_binary, average='binary', zero_division=0)

                print(f"  GT positives: {np.sum(gt_binary)}, Pred positives: {np.sum(pred_binary)}")
                print(f"  Metrics - IoU: {iou:.4f}, Acc: {acc:.4f}, Prec: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")

                metrics = {
                    f"mIoU_{class_name}": iou,
                    f"Accuracy_{class_name}": acc,
                    f"Precision_{class_name}": prec,
                    f"Recall_{class_name}": rec,
                    f"F1 Score_{class_name}": f1
                }

        class_metrics.update(metrics)

    return class_metrics


def save_metrics_to_csv_classes(filename, video_name, fusion_threshold, avg_metrics_normal, avg_metrics_overwritten, num_frames):
    if num_frames == 0:
        print("No valid frames processed. Skipping CSV writing.")
        return

    if not filename.endswith("_classes.csv"):
        base, ext = os.path.splitext(filename)
        filename = f"{base}_classes.csv"

    class_names = list(CITYSCAPES_CLASS_NAMES.values())
    metric_types = ["mIoU", "Accuracy", "Precision", "Recall", "F1 Score"]

    csv_headers = ["Video Name", "Fusion Threshold"]
    csv_values = [video_name, fusion_threshold]

    for metric in metric_types:
        for class_name in class_names:
            key = f"{metric}_{class_name}"
            csv_headers.append(f"Normal {metric} {class_name}")
            csv_headers.append(f"Overwritten {metric} {class_name}")
            csv_values.append(avg_metrics_normal.get(key, 0.0))
            csv_values.append(avg_metrics_overwritten.get(key, 0.0))

    file_exists = os.path.isfile(filename)
    with open(filename, mode="a", newline="") as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(csv_headers)
        writer.writerow(csv_values)

    print(f"\nMetrics appended to {filename}")


# Example main evaluation loop that uses the above functions
def evaluate_video(pred_masks_normal, pred_masks_overwritten, gt_masks, video_name, fusion_threshold, output_csv_path):
    sum_metrics_normal = {}
    sum_metrics_overwritten = {}
    frame_count = 0

    for pred_normal, pred_overwritten, gt_mask in zip(pred_masks_normal, pred_masks_overwritten, gt_masks):
        metrics_normal = calculate_metrics_classes(pred_normal, gt_mask, CITYSCAPES_TO_VSPW_MAPPING, VSPW_TO_CITYSCAPES_MAPPING)
        metrics_overwritten = calculate_metrics_classes(pred_overwritten, gt_mask, CITYSCAPES_TO_VSPW_MAPPING, VSPW_TO_CITYSCAPES_MAPPING)

        for k, v in metrics_normal.items():
            sum_metrics_normal[k] = sum_metrics_normal.get(k, 0.0) + v
        for k, v in metrics_overwritten.items():
            sum_metrics_overwritten[k] = sum_metrics_overwritten.get(k, 0.0) + v

        frame_count += 1

    avg_metrics_normal = {k: v / frame_count for k, v in sum_metrics_normal.items()}
    avg_metrics_overwritten = {k: v / frame_count for k, v in sum_metrics_overwritten.items()}

    print(f"\nAverage Metrics for Normal Segmentation at {frame_count} frames:")
    print(avg_metrics_normal)
    print(f"\nAverage Metrics for Overwritten Segmentation at {frame_count} frames:")
    print(avg_metrics_overwritten)

    save_metrics_to_csv_classes(output_csv_path, video_name, fusion_threshold,
                                 avg_metrics_normal, avg_metrics_overwritten, frame_count)
