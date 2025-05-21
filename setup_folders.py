import os

def create_folder_structure():
    """
    Create the necessary folder structure for the PCOS Meal Plan Generator
    """
    folders = [
        'data/raw',           # For storing raw docx files
        'data/processed',     # For storing processed data
        'models',            # For storing trained models
        'logs',              # For storing training logs
        'static',            # For storing static web files
        'static/css',        # For CSS files
        'static/js',         # For JavaScript files
        'templates'          # For HTML templates
    ]
    
    for folder in folders:
        os.makedirs(folder, exist_ok=True)
        print(f"Created folder: {folder}")

if __name__ == "__main__":
    create_folder_structure() 