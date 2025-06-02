import os
import sys
from collections import defaultdict
import cv2
import numpy as np
import torch
import time
from datetime import datetime
from PIL import Image
from ultralytics import YOLO
from mmengine.model.utils import revert_sync_batchnorm
from mmseg.apis import init_model, init_model_classes_palette, inference_model, show_result_pyplot, show_result_pyplot_with_timer
from IPython.display import display, clear_output
from fusionVSPWutils_classes import *

# Check if VIDEO_NAME is passed as an argument
if len(sys.argv) != 2:
    print("Usage: python script.py <VIDEO_NAME>")
    sys.exit(1)

VIDEO_NAME = sys.argv[1]

# Constants and file paths
FRAME_WIDTH_CUSTOM = 1920
FRAME_HEIGHT_CUSTOM = 540
FUSION_THRESHOLD = 0

BASE_DIR = "/content/drive/MyDrive/ureca/mmsegmentation"
DATA_DIR = f"{BASE_DIR}/data/VSPW/data"
VIDEO_DIR = os.path.join(DATA_DIR, VIDEO_NAME)
IMAGE_DIR = os.path.join(VIDEO_DIR, "origin")
MASK_DIR = os.path.join(VIDEO_DIR, "mask")
VIDEO_NAME = os.path.basename(os.path.dirname(IMAGE_DIR))
OUTPUT_PATH = f"{BASE_DIR}/demo/output/fusionoutputs/overwriting_semantic_deeplabv3plus_{VIDEO_NAME}_<{FUSION_THRESHOLD}.mp4"
CONFIG_FILE = f'{BASE_DIR}/configs/deeplabv3plus/deeplabv3plus_r101-d8_4xb2-80k_cityscapes-512x1024.py'
CHECKPOINT_FILE = f'{BASE_DIR}/checkpoints/deeplabv3plus/deeplabv3plus_r101-d8_512x1024_80k_cityscapes_20200606_114143-068fcfe9.pth'
OVERWRITE_TIMES_FILE = f'{BASE_DIR}/demo/overwrite_times_deeplabv3plus_{VIDEO_NAME}_<{FUSION_THRESHOLD}_classes.txt'

# Flags and options
PRINT_TIMINGS = True  # Set this to False to disable timing prints
SAVE_VIDEO = False
WRITE_TXT = False
METRICS_FOR_CLASSES = True
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
if torch.cuda.is_available():
    print("USING CUDA")
    
# Get sorted list of image files
image_files = sorted([f for f in os.listdir(IMAGE_DIR) if f.endswith(('.png', '.jpg'))])
image_paths = [os.path.join(IMAGE_DIR, f) for f in image_files]

if not image_paths:
    print("Error: No images found in the directory.")
    exit()

# Generate corresponding mask paths
mask_paths = []
for img_file in image_files:
    for ext in ('.png', '.jpg'):  # Try both extensions for mask files
        mask_path = os.path.join(MASK_DIR, os.path.splitext(img_file)[0] + ext)
        if os.path.exists(mask_path):
            mask_paths.append(mask_path)
            break
    else:
        print(f"Warning: No mask found for {img_file}. Skipping this image.")
        image_paths.remove(os.path.join(IMAGE_DIR, img_file))  # Remove images without masks

if not image_paths or not mask_paths:
    print("Error: No valid image-mask pairs found.")
    exit()

# Load models
model_yolo = YOLO("yolo11n.pt")
model_seg, semseg_classes, semseg_palette = init_model_classes_palette(CONFIG_FILE, CHECKPOINT_FILE, device=DEVICE)

# Mapping between YOLO and segmentation class names
model_yolo_names_dict = model_yolo.names
model_seg_names_dict = {i: class_name for i, class_name in enumerate(semseg_classes)}
print(model_yolo_names_dict)
print(model_seg_names_dict)
yolo_to_seg_dict, seg_to_yolo_dict = {}, {}
for yolo_id, yolo_name in model_yolo_names_dict.items():
    for seg_id, seg_name in model_seg_names_dict.items():
        if yolo_name == seg_name:
            yolo_to_seg_dict[yolo_id] = seg_id
            seg_to_yolo_dict[seg_id] = yolo_id
            break
print("YOLO to Segmentation Mapping:", yolo_to_seg_dict)
print("Segmentation to YOLO Mapping:", seg_to_yolo_dict)

# If not using CUDA, revert sync batchnorm
if not torch.cuda.is_available():
    model_seg = revert_sync_batchnorm(model_seg)

# Store track history
track_history = defaultdict(lambda: [])

object_masks = {}
num_frames = 0
metric_keys = ["mIoU", "Accuracy", "Precision", "Recall", "F1 Score"]
total_metrics_normal = {key: 0.0 for key in metric_keys}
total_metrics_overwritten = {key: 0.0 for key in metric_keys}
total_metrics_normal = defaultdict(float)
total_metrics_overwritten = defaultdict(float)

average_metrics_normal_classes = initialise_metrics_classes()
average_metrics_overwritten_classes = initialise_metrics_classes()

for image_path, mask_path in zip(image_paths, mask_paths):

    frame = cv2.imread(image_path)

    if frame is None:
        print(f"Error reading image: {image_path}")
        continue
    
    num_frames += 1

    # Step 1: Run YOLO object detection and tracking
    results_yolo = model_yolo.track(frame, persist=True, tracker="botsort.yaml")

    # Step 2: Run semantic segmentation (DeepLabV3+)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    start_time = time.time()
    result_seg = inference_model(model_seg, frame_rgb)

    gt_mask = Image.open(mask_path)
    gt_mask_array = np.array(gt_mask)

    inference_time = time.time() - start_time
    # if PRINT_TIMINGS:
    #     print(f"Inference Time for segmentation: {inference_time:.4f} seconds")


    # Step 3: Process the YOLO results for object tracking
    if results_yolo[0].boxes is not None and results_yolo[0].boxes.id is not None:
        boxes = results_yolo[0].boxes.xywh.cpu()
        track_ids = results_yolo[0].boxes.id.int().cpu().tolist()
        
        ### Cropped masks
        start = time.time()

        for track_id, (box, yolo_class_id) in zip(track_ids, zip(boxes, results_yolo[0].boxes.cls.tolist())):
            # Remove object masks where track_id is older than track_id - 500
            # keys_to_remove = [key for key in object_masks.keys() if key < track_id - 200]
            # for key in keys_to_remove:
            #     del object_masks[key]

            # Check if this class ID is in the yolo_to_seg_dict
            if yolo_class_id in yolo_to_seg_dict:  # No need for .item() here
                seg_class_id = yolo_to_seg_dict[int(yolo_class_id)]
                # Convert the box coordinates to integers (x1, y1, x2, y2)
                # x1, y1, x2, y2 = map(int, box)
                
                mid_x, mid_y, width, height = box

                # Convert (mid_x, mid_y, width, height) to (x1, y1, x2, y2)
                x1 = mid_x - width / 2
                y1 = mid_y - height / 2
                x2 = mid_x + width / 2
                y2 = mid_y + height / 2

                # Ensure the bounding box is valid (i.e., within image bounds)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
                
                x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))

                # Crop the image using the bounding box coordinates (top-left to bottom-right)
                cropped_image = frame[y1:y2, x1:x2]

                # MASK OVERLAY
                mask = result_seg.pred_sem_seg.data[0].cpu().numpy()
                cropped_mask = mask[y1:y2, x1:x2]
                mask_region = mask[y1:y2, x1:x2]
                colored_mask = np.zeros((y2 - y1, x2 - x1, 3), dtype=np.uint8)
                for class_id in np.unique(mask_region):
                    # if class_id >= len(semseg_palette):
                    #     continue
                    color = semseg_palette[int(class_id)]
                    colored_mask[mask_region == class_id] = color
                cropped_image = cv2.addWeighted(cropped_image, 1 - 0.5, colored_mask, 0.5, 0)


                # Check if the cropped image is empty
                if cropped_image.size == 0:
                    print(f"Empty cropped image for box: ({x1}, {y1}, {x2}, {y2}), skipping.")
                    continue

                # Calculate the percentage match (matching pixels between the mask and the object class)
                matching_pixels = (mask_region == seg_class_id).sum()
                total_pixels = mask_region.size
                percentage_match = (matching_pixels / total_pixels) * 100

                # Get the previous percentage match, default to None if not found
                if track_id in object_masks:
                    last_percentage_match = object_masks[track_id]['last_percentage_match']
                    average_percentage = object_masks[track_id]['average_percentage']

                    percentage_change_from_last = 100 * (percentage_match - last_percentage_match) / last_percentage_match
                    percentage_change_from_average = 100 * (percentage_match - average_percentage) / average_percentage
                    
                    if percentage_change_from_last < FUSION_THRESHOLD and percentage_change_from_average < FUSION_THRESHOLD:
                        # Unacceptable mask
                        print("----UNACCEPTED----")
                        object_masks[track_id]['overwrite'] = True
                        object_masks[track_id].update({
                            'percentage_change_from_last': percentage_change_from_last,
                            'percentage_change_from_average': percentage_change_from_average
                        })
                    
                    else:
                        # Update the count and calculate the average percentage match
                        print("----ADDING EXISTING----")
                        count = object_masks[track_id]['count'] + 1
                        average_percentage = ((object_masks[track_id]['average_percentage'] * object_masks[track_id]['count']) + percentage_match) / count

                        # Update the object_masks dictionary with new count and average_percentage
                        # cropped_mask.fill(22)
                        object_masks[track_id].update({
                            'mask': cropped_mask,
                            'box': [x1, y1, x2, y2],
                            'last_percentage_match': percentage_match,
                            'count': count,
                            'percentage_change_from_last': percentage_change_from_last,
                            'percentage_change_from_average': percentage_change_from_average,
                            'average_percentage': average_percentage,
                            'overwrite': False
                        })

                else:
                    # For the first frame of this track_id
                    print("----ADDING NEW----")
                    average_percentage = percentage_match
                    # cropped_mask.fill(22)
                    object_masks[track_id] = {
                        'mask': cropped_mask,
                        'box': [x1, y1, x2, y2],
                        'last_percentage_match': percentage_match,
                        'percentage_change_from_last': 0,
                        'percentage_change_from_average': 0,
                        'count': 1,
                        'average_percentage': average_percentage,
                        'overwrite': False
                    }

                # print("\n----OBJECT MASKS----")
                # print(object_masks)
                # print("----END OBJECT MASKS----\n")

                # Save the cropped image with a unique filename
                # output_filename = f"fusion_output/crop_{yolo_class_id}_{track_id}.png"
                # cv2.imwrite(output_filename, cropped_image)
                # print(f"Saved cropped image: {output_filename}")

        # End the timer and calculate the elapsed time
        end = time.time()
        elapsed_time = end - start
        # print(f"Time taken for cropped masks: {elapsed_time:.2f} seconds")


        ### End Cropped masks

        objtrack_annotated_frame = results_yolo[0].plot()  # Annotate frame with YOLO (bounding boxes and tracks)

        # Plot the tracks from BotSort
        for box, track_id in zip(boxes, track_ids):
            x, y, w, h = box
            track = track_history[track_id]
            track.append((float(x), float(y)))  # track center point
            if len(track) > 30:
                track.pop(0)
            points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(objtrack_annotated_frame, [points], isClosed=False, color=(230, 230, 230), thickness=3)

    else:
        objtrack_annotated_frame = frame  # No objects detected, keep original frame

    # Step 4: Overlay the semantic segmentation result

    # vis_result = show_result_pyplot(model_seg, frame_rgb, result_seg, show=False, with_labels=False)
    # vis_result_bgr = cv2.cvtColor(vis_result, cv2.COLOR_RGB2BGR)
    
    # Assuming 'full_mask' is the original mask and 'full_mask_overwritten' is the modified copy
    full_mask = result_seg.pred_sem_seg.data[0].cpu().numpy()
    full_mask_overwritten = full_mask.copy()  # This will keep track of the overwritten masks

    # Open the file for appending the overwrite times only if WRITE_TXT is True
    if WRITE_TXT:
        f = open(OVERWRITE_TIMES_FILE, 'a')

    if results_yolo[0].boxes is not None and results_yolo[0].boxes.id is not None:
        print("track_ids found")
        for track_id, (box, yolo_class_id) in zip(track_ids, zip(boxes, results_yolo[0].boxes.cls.tolist())):
            
            if track_id in object_masks and yolo_class_id in yolo_to_seg_dict:
                object_mask = object_masks[track_id]

                if object_mask['overwrite']:
                    seg_class_id = yolo_to_seg_dict[yolo_class_id]
                    old_x1, old_y1, old_x2, old_y2 = map(int, object_mask['box'])

                    # Assuming box is in the format (mid_x, mid_y, width, height)
                    new_mid_x, new_mid_y, new_width, new_height = box
                    new_x1 = new_mid_x - new_width / 2
                    new_y1 = new_mid_y - new_height / 2
                    new_x2 = new_mid_x + new_width / 2
                    new_y2 = new_mid_y + new_height / 2

                    new_mid_x, new_mid_y, new_width, new_height = map(int, [new_mid_x, new_mid_y, new_width, new_height])
                    new_x1, new_y1, new_x2, new_y2 = map(int, [new_x1, new_y1, new_x2, new_y2])

                    overwriting_mask = object_mask['mask']

                    # Get the region to overwrite
                    overwriting_region = full_mask_overwritten[new_y1:new_y2, new_x1:new_x2]
                    region_height, region_width = overwriting_region.shape

                    if region_height > 0 and region_width > 0:
                        # Resize the overwriting mask to match the region size
                        reshaped_overwriting_mask = cv2.resize(overwriting_mask, (region_width, region_height), interpolation=cv2.INTER_NEAREST_EXACT)

                        # Create the boolean mask based on class id
                        boolean_mask = (reshaped_overwriting_mask == seg_class_id)

                        # Apply reshaped mask to the relevant region
                        overwriting_region[boolean_mask] = reshaped_overwriting_mask[boolean_mask]

                        # Assign the updated overwriting region back to the full mask
                        full_mask_overwritten[new_y1:new_y2, new_x1:new_x2] = overwriting_region
    else:
        print("track_ids not found")
    # Close file only if it was opened
    if WRITE_TXT:
        f.close()

    # Compute metrics for normal and overwritten masks
    if METRICS_FOR_CLASSES:
        average_metrics_normal_classes = calculate_average_metrics_classes(average_metrics_normal_classes, full_mask, gt_mask_array, CITYSCAPES_TO_VSPW_MAPPING, VSPW_TO_CITYSCAPES_MAPPING)
        average_metrics_overwritten_classes = calculate_average_metrics_classes(average_metrics_overwritten_classes, full_mask_overwritten, gt_mask_array, CITYSCAPES_TO_VSPW_MAPPING, VSPW_TO_CITYSCAPES_MAPPING)
        avg_metrics_normal = average_metrics_normal_classes
        avg_metrics_overwritten = average_metrics_overwritten_classes

    else:
        metrics_normal = calculate_metrics(full_mask, gt_mask_array, CITYSCAPES_TO_VSPW_MAPPING, VSPW_TO_CITYSCAPES_MAPPING)
        metrics_overwritten = calculate_metrics(full_mask_overwritten, gt_mask_array, CITYSCAPES_TO_VSPW_MAPPING, VSPW_TO_CITYSCAPES_MAPPING)

        # Accumulate metrics
        for key in total_metrics_normal:
            total_metrics_normal[key] += metrics_normal[key]
            total_metrics_overwritten[key] += metrics_overwritten[key]

        # Compute averages if at least one valid frame was processed
        if num_frames > 0:
            avg_metrics_normal = {key: total / num_frames for key, total in total_metrics_normal.items()}
            avg_metrics_overwritten = {key: total / num_frames for key, total in total_metrics_overwritten.items()}
    
    # Print metrics
    if num_frames > 0:
        print(f"\nAverage Metrics for Normal Segmentation at {num_frames} frames:")
        print(avg_metrics_normal)

        print(f"\nAverage Metrics for Overwritten Segmentation at {num_frames} frames:")
        print(avg_metrics_overwritten)
    else:
        print("No valid frames processed.")
        
    print()
    print(f"Video {VIDEO_NAME}, frame {num_frames}")
    print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print()

    # Break if 'q' is pressed
    # Removed this due to being on Jupyter Notebook
    # if cv2.waitKey(1) & 0xFF == ord("q"):
    #     break

if METRICS_FOR_CLASSES:
    save_metrics_to_csv_classes(f"VSPW_segmentation_metrics_mobilenetv3_{FUSION_THRESHOLD}.csv", VIDEO_NAME, FUSION_THRESHOLD, avg_metrics_normal, avg_metrics_overwritten, num_frames)
else:
    save_metrics_to_csv(f"VSPW_segmentation_metrics_mobilenetv3_{FUSION_THRESHOLD}.csv", VIDEO_NAME, FUSION_THRESHOLD, avg_metrics_normal, avg_metrics_overwritten, num_frames)
# Release resources
cv2.destroyAllWindows()
object_masks.clear()

