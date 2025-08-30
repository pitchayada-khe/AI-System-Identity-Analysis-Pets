import os

label_folder = 'D:/CE PROJECT/models/dataset2/valid/labels'

class_mapping = {
    0: 15,
    1: 16,
    2: 80
}

for filename in os.listdir(label_folder):
    if filename.endswith('.txt'):
        file_path = os.path.join(label_folder, filename)
        
        with open(file_path, 'r') as f:
            lines = f.readlines()

        new_lines = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 5:
                old_class = int(parts[0])
                if old_class in class_mapping:
                    new_class = class_mapping[old_class]
                    parts[0] = str(new_class)
                    new_lines.append(' '.join(parts) + '\n')

        with open(file_path, 'w') as f:
            f.writelines(new_lines)
    
    print(f"Mapping complete: {filename}")
