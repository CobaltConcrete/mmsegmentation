input_path = "scripts.txt"       # Your original file
output_path = "kaggle_scripts.txt" # New file with '!' added

with open(input_path, "r") as infile, open(output_path, "w") as outfile:
    for line in infile:
        line = line.strip()
        if line:  # skip empty lines
            outfile.write(f"!{line}\n")
