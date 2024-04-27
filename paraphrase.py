import os
from docx import Document

def create_docx_from_folder(folder_path, output_file):
    # Create a new Word document
    doc = Document()

    # Iterate through each file in the folder
    for filename in sorted(os.listdir(folder_path)):
        if filename.endswith(".txt"):
            # Read the text file
            with open(os.path.join(folder_path, filename), "r", encoding="utf-8") as file:
                text = file.read()
            
            # Add text to the document
            doc.add_paragraph(text)

    # Save the document
    doc.save(output_file)
    print(f"Word document saved as '{output_file}'")

# Example usage:
folder_path = "train_data"
output_file = "base_data.docx"
create_docx_from_folder(folder_path, output_file)
