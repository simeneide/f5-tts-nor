import json
import pandas as pd
import os
from pathlib import Path
import torchaudio
import torch.nn.functional as F
import shutil
import argparse

def create_directories():
    """Create necessary directories if they don't exist"""
    Path("/mnt/llm/datasets/nst_ostlandsk/wavs").mkdir(parents=True, exist_ok=True)

def load_metadata(directory):
    """Load metadata from json files"""
    directory = Path(directory)
    all_metadata = []
    
    metadata_files = list(directory.glob('*tar_metadata.json'))
    print(f"Found {len(metadata_files)} metadata files in {directory}")
    
    for metadata_file in metadata_files:
        try:
            with open(metadata_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        metadata = json.loads(line.strip())
                        # Only append if it has the required fields
                        if 'Region_of_Youth' in metadata:
                            all_metadata.append(metadata)
                    except json.JSONDecodeError as e:
                        print(f"Error parsing line in {metadata_file.name}: {str(e)}")
                        continue
            print(f"Loaded entries from {metadata_file.name}")
        except Exception as e:
            print(f"Error loading {metadata_file}: {str(e)}")
    
    df = pd.DataFrame(all_metadata)
    print(f"Total entries loaded: {len(df)}")
    
    # Print available columns
    print("Available columns:", df.columns.tolist())
    
    return df

def process_audio(input_path, output_path, target_sr=24000):
    """Load and resample audio to 24kHz"""
    waveform, sr = torchaudio.load(input_path)
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(sr, target_sr)
        waveform = resampler(waveform)
    torchaudio.save(output_path, waveform, target_sr)

def main(input_dataset, output_dir):
    # Create directories
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Define østlandsk regions
    ostlandsk_regions = {'Oslo-området', 'Hedmark og Oppland', 'Ytre Oslofjord'}
    
    # Process train and test data
    metadata_rows = []
    for data_dir in ['train', 'test']:
        print(f"Processing {data_dir}...")
        full_data_dir = Path(input_dataset) / data_dir
        df = load_metadata(full_data_dir)
        
        # Filter for østlandsk regions
        df_ostlandsk = df[df['Region_of_Youth'].isin(ostlandsk_regions)]
        print(f"Found {len(df_ostlandsk)} entries from østlandsk regions")
        
        for _, row in df_ostlandsk.iterrows():
            pid = row['pid']
            wav_file = row['file']
            mp3_file = f"{pid}_{wav_file}".replace('.wav', '.mp3')
            input_path = full_data_dir / mp3_file
            
            if input_path.exists():
                # Create output filename
                output_filename = f"{pid}_{wav_file}"
                output_path = output_dir / output_filename
                
                # Convert and save audio
                try:
                    process_audio(input_path, output_path)
                    
                    # Add to metadata
                    metadata_rows.append({
                        'path': str(output_path.absolute()),
                        'text': row['text']
                    })
                    print(f"Processed: {output_filename}")
                except Exception as e:
                    print(f"Error processing {input_path}: {str(e)}")
            else:
                print(f"File not found: {input_path}")
    
    # Create and save metadata.csv
    metadata_df = pd.DataFrame(metadata_rows)
    metadata_df.to_csv(output_dir / 'metadata.csv', 
                      sep='|', 
                      index=False, 
                      header=False)
    print(f"Created metadata.csv with {len(metadata_df)} entries")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create NST dataset with østlandsk dialect')
    parser.add_argument('--input-dataset', required=True,
                      help='Base directory containing train/ and test/ folders')
    parser.add_argument('--output-dir', required=True,
                      help='Output directory for processed dataset')
    
    args = parser.parse_args()
    main(args.input_dataset, args.output_dir)
