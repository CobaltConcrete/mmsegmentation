import os
from collections import defaultdict
import cv2
import numpy as np
import torch
import time
from ultralytics import YOLO
from mmengine.model.utils import revert_sync_batchnorm
from mmseg.apis import init_model, init_model_classes_palette, inference_model, show_result_pyplot, show_result_pyplot_with_timer
from IPython.display import display, clear_output

# Constants and file paths
FRAME_WIDTH_CUSTOM = 1920
FRAME_HEIGHT_CUSTOM = 540
FUSION_THRESHOLD = 0

VIDEO_PATH = '/home/makers/Documents/mmsegmentation/data/custom/Photos/20241223_190939.mp4'
VIDEO_NAME = os.path.splitext(os.path.basename(VIDEO_PATH))[0]
OUTPUT_PATH = f"/home/makers/Documents/mmsegmentation/demo/output/fusionoutputs/overwriting_semantic_deeplabv3plus_{VIDEO_NAME}_<{FUSION_THRESHOLD}.mp4"
CONFIG_FILE = '/home/makers/Documents/mmsegmentation/configs/deeplabv3plus/deeplabv3plus_r101-d8_4xb2-80k_cityscapes-512x1024.py'
CHECKPOINT_FILE = '/home/makers/Documents/mmsegmentation/checkpoints/deeplabv3plus/deeplabv3plus_r101-d8_512x1024_80k_cityscapes_20200606_114143-068fcfe9.pth'
OVERWRITE_TIMES_FILE = f'/home/makers/Documents/mmsegmentation/demo/overwrite_times_deeplabv3plus_{VIDEO_NAME}_<{FUSION_THRESHOLD}.txt'

# Flags and options
PRINT_TIMINGS = True  # Set this to False to disable timing prints
SAVE_VIDEO = False
WRITE_TXT = False
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'



# Initialize VideoCapture for reading the video
cap = cv2.VideoCapture(VIDEO_PATH)

# Check if the video was opened successfully
if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

# Get video properties (FPS and frame size)
fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Create a VideoWriter object to save the output video
if SAVE_VIDEO:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Codec for saving video
    out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (FRAME_WIDTH_CUSTOM, FRAME_HEIGHT_CUSTOM))

# Initialize models
# Load YOLOv5 model
model_yolo = YOLO("yolo11n.pt")

# Initialize semantic segmentation model and its metadata (classes, palette)
model_seg, semseg_classes, semseg_palette = init_model_classes_palette(CONFIG_FILE, CHECKPOINT_FILE, device=DEVICE)

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
print(model_yolo_names_dict)

# If not using CUDA, revert sync batchnorm
if not torch.cuda.is_available():
    model_seg = revert_sync_batchnorm(model_seg)

# Store track history
track_history = defaultdict(lambda: [])

object_masks = {}

while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        
        # Step 1: Run YOLO object detection and tracking
        results_yolo = model_yolo.track(frame, persist=True, tracker="botsort.yaml")

        # Step 2: Run semantic segmentation (DeepLabV3+)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        start_time = time.time()

        result_seg = inference_model(model_seg, frame_rgb)

        if False:
            print("\n----RESULT SEG----\n")
            print(result_seg.pred_sem_seg.data[0], "\n")
            print(result_seg.pred_sem_seg.data[0].size())
            print("\n----END RESULT SEG----\n")
            print("\n----SEMSEG CLASSES----\n")
            print(semseg_classes)
            print(type(semseg_classes))
            print(len(semseg_classes))
            print("\n----END SEMSEG CLASSES----\n")
            print("\n----SEMSEG PALETTE----\n")
            print(semseg_palette)
            print(type(semseg_palette))
            print(len(semseg_palette))
            print("\n----END SEMSEG PALETTE----\n")

            print("yolotosegdict")
            print(yolo_to_seg_dict)
            print(seg_to_yolo_dict)

        inference_time = time.time() - start_time
        # if PRINT_TIMINGS:
        #     print(f"Inference Time for segmentation: {inference_time:.4f} seconds")


        # Step 3: Process the YOLO results for object tracking
        if results_yolo[0].boxes is not None and results_yolo[0].boxes.id is not None:
            boxes = results_yolo[0].boxes.xywh.cpu()
            track_ids = results_yolo[0].boxes.id.int().cpu().tolist()

            if False:
                print("\n----TRACKER results_yolo[0].names----\n")
                print(results_yolo[0].names)
                print(type(results_yolo[0].names))
                print(len(results_yolo[0].names))
                print("\n----END TRACKER results_yolo[0].names----\n")

                print("\n----TRACKER results_yolo[0].boxes.cls----\n")
                print(results_yolo[0].boxes.cls)
                print(type(results_yolo[0].boxes.cls))
                print(len(results_yolo[0].boxes.cls))
                print("\n----END TRACKER results_yolo[0].boxes.cls----\n")

                print("\n----TRACKER tracker_namelist----\n")
                class_ids = results_yolo[0].boxes.cls.int().cpu().tolist()
                tracker_namelist = [results_yolo[0].names[int(class_id)] for class_id in class_ids]
                print(tracker_namelist)
                print(type(tracker_namelist))
                print(len(tracker_namelist))
                print("\n----END TRACKER tracker_namelist----\n")


                print("\n----TRACKER results_yolo[0].boxes.conf----\n")
                print(results_yolo[0].boxes.conf)
                print(type(results_yolo[0].boxes.conf))
                print(len(results_yolo[0].boxes.conf))
                print("\n----END TRACKER results_yolo[0].conf----\n")

                print("\n----TRACKER BOXES----\n")
                print(boxes)
                print(type(boxes))
                print(len(boxes))
                print("\n----END TRACKER BOXES----\n")

                print("\n----TRACKER IDS----\n")
                print(track_ids)
                print(type(track_ids))
                print(len(track_ids))
                print("\n----END TRACKER IDS----\n")
            
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
                    print("seg_class_id     =", seg_class_id)
                    print("yolo_class_id    =", yolo_class_id)
                    print("yolo_to_seg_dict =", yolo_to_seg_dict)
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
                    output_filename = f"fusion_output/crop_{yolo_class_id}_{track_id}.png"
                    cv2.imwrite(output_filename, cropped_image)
                    print(f"Saved cropped image: {output_filename}")

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

        # Open the file for appending the overwrite times
        with open(OVERWRITE_TIMES_FILE, 'a') as f:

            for track_id, (box, yolo_class_id) in zip(track_ids, zip(boxes, results_yolo[0].boxes.cls.tolist())):
                if track_id in object_masks and yolo_class_id in yolo_to_seg_dict:
                    object_mask = object_masks[track_id]

                    if object_mask['overwrite'] == True:
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

                            # Now fill the region with a specific value (e.g., 22 for black)
                            # overwriting_region.fill(22)

                            # Assign the updated overwriting region back to the full mask
                            full_mask_overwritten[new_y1:new_y2, new_x1:new_x2] = overwriting_region

                            # Log time and other data
                            video_time_ms = cap.get(cv2.CAP_PROP_POS_MSEC)  # Get time in milliseconds
                            video_time_sec = video_time_ms / 1000  # Convert to seconds

                            log_message = f"Track ID: {track_id}, YOLO Class ID: {yolo_class_id}, Overwritten at: (Video Time: {video_time_sec:.2f}s), Percentage change from last: {object_mask['percentage_change_from_last']}, Percentage change from average: {object_mask['percentage_change_from_average']}\n"
                            f.write(log_message)  # Log to file
                            print(f"Mask overwritten at (Video Time: {video_time_sec:.2f}s). Recorded time.")


        # Annotate Original semseg frame
        frame_height, frame_width = frame.shape[:2]
        full_colored_mask = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
        for class_id in np.unique(full_mask):
            if class_id >= len(semseg_palette):
                continue
            color = semseg_palette[int(class_id)]
            full_colored_mask[full_mask == class_id] = color
        alpha = .9
        semseg_annotated_frame = cv2.addWeighted(frame, 1 - alpha, full_colored_mask, alpha, 0)

        # Annotate Overwritten semseg frame
        full_colored_mask_overwritten = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
        for class_id in np.unique(full_mask_overwritten):
            if class_id >= len(semseg_palette):
                continue
            color = semseg_palette[int(class_id)]
            full_colored_mask_overwritten[full_mask_overwritten == class_id] = color
        alpha = 0.9
        semseg_annotated_frame_overwritten = cv2.addWeighted(frame, 1 - alpha, full_colored_mask_overwritten, alpha, 0)



        # Combine annotated frame and semantic segmentation result
        combined_frame = cv2.hconcat([semseg_annotated_frame_overwritten, semseg_annotated_frame])

        # Resize the combined frame for better display
        height, width = combined_frame.shape[:2]
        combined_frame_resized = cv2.resize(combined_frame, (width // 2, height // 2))

        # Show the combined frame
        cv2.imshow('YOLO + Semantic Segmentation (Resized)', combined_frame_resized)

        # Write to output video
        if SAVE_VIDEO:
            out.write(combined_frame_resized)

        # Get the size of the combined frame (height, width, channels)
        height, width, channels = combined_frame_resized.shape

        # Print the size
        print(f"Combined Frame Resized Size: Height={height}, Width={width}, Channels={channels}")


        # Break if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if video ends
        break

# Release resources
cap.release()
if SAVE_VIDEO:
    out.release()
cv2.destroyAllWindows()
