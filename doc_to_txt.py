import os
from docx import Document

def create_txt_files_from_docx(docx_file, output_folder):
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Load the Word document
    doc = Document(docx_file)
    
    # Initialize index for filenames
    file_index = 0
    
    # Iterate through paragraphs
    for para in doc.paragraphs:
        # If paragraph is empty, skip to the next paragraph
        if not para.text.strip():
            continue
        
        # Create a new text file for each paragraph
        filename = os.path.join(output_folder, f"FID-{file_index}-5.txt")
        with open(filename, "w", encoding="utf-8") as file:
            file.write(para.text)
        
        file_index += 1

    print(f"Text files created in '{output_folder}'")

# Example usage:
docx_file = "doc_correct.docx"
output_folder = "generated_data/paraphrase_data"
create_txt_files_from_docx(docx_file, output_folder)
