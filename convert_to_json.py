import json
from pathlib import Path

def main():
    print("Converting 3 files to JSON...")
    data_dir = Path('./data')
    
    if not data_dir.exists():
        print("Error: Could not find the 'data' folder.")
        return

    # Grab the first 3 markdown files
    md_files = list(data_dir.glob('*.md'))[:3]
    
    if not md_files:
        print("No .md files found to convert!")
        return
    
    for filepath in md_files:
        # Read the original text
        text_content = filepath.read_text(encoding='utf-8', errors='ignore')
        
        # Create a structured dictionary
        json_data = {
            "source_file": filepath.name,
            "document_type": "Kvant Internal Record",
            "content": text_content
        }
        
        # Save as a new .json file
        json_filepath = filepath.with_suffix('.json')
        with open(json_filepath, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=4, ensure_ascii=False)
            
        print(f"Created: {json_filepath.name}")
        
        # Delete the original .md file so we don't have duplicates
        filepath.unlink()

    print("Done! You now have shiny new JSON files in your data folder.")

if __name__ == "__main__":
    main()