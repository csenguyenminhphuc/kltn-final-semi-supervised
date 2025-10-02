import os
import shutil

# Create web folder structure
web_structure = {
    'web': {
        'static': {
            'css': {},
            'js': {},
            'uploads': {}
        },
        'templates': {},
        'models': {},
        'app.py': None,
        'requirements.txt': None
    }
}

def create_folder_structure(base_path, structure):
    for name, content in structure.items():
        path = os.path.join(base_path, name)
        if content is None:
            # This is a file placeholder
            continue
        else:
            # This is a folder
            os.makedirs(path, exist_ok=True)
            if isinstance(content, dict):
                create_folder_structure(path, content)

# Create the web project structure
base_dir = '/home/coder/trong/KLTN_SEMI'
create_folder_structure(base_dir, web_structure)

print("Web folder structure created successfully!")
print("Structure:")
for root, dirs, files in os.walk(os.path.join(base_dir, 'web')):
    level = root.replace(base_dir, '').count(os.sep)
    indent = ' ' * 2 * level
    print(f'{indent}{os.path.basename(root)}/')
    subindent = ' ' * 2 * (level + 1)
    for file in files:
        print(f'{subindent}{file}')