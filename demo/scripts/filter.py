csv_path = '/home/r13qingrong/Projects/URECA/mmsegmentation/demo/VSPW_segmentation_metrics_mobilenetv3_0_classes.csv'
txt_path = '/home/r13qingrong/Projects/URECA/mmsegmentation/demo/VSPWvideofiles.txt'
output_path = '/home/r13qingrong/Projects/URECA/mmsegmentation/demo/VSPWvideofiles_filtered.txt'

# Read video names from CSV (first column)
with open(csv_path, 'r') as f:
    csv_lines = f.readlines()
csv_video_names = {line.split(',')[0] for line in csv_lines[1:]}  # Skip header

# Filter lines in TXT that are not in CSV
with open(txt_path, 'r') as f:
    txt_lines = f.readlines()

filtered_lines = [line for line in txt_lines if line.strip() not in csv_video_names]

# Write to new TXT file
with open(output_path, 'w') as f:
    f.writelines(filtered_lines)

print(f"✅ Done: Filtered list written to {output_path}")
