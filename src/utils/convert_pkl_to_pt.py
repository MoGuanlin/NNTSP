
import os
import argparse
import pickle
import torch
from types import SimpleNamespace
from tqdm import tqdm
import sys

# Add project root to path to import from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.dataprep.dataset import consolidate_data_list

def convert_pkl_to_pt(input_path, output_path):
    print(f"Loading {input_path}...")
    with open(input_path, 'rb') as f:
        data = pickle.load(f)
    
    # data is a list of instances, each instance is a list of [x, y] coordinates
    print(f"Converting {len(data)} instances...")
    instances = []
    for instance in tqdm(data):
        # Scale back to 10000.0 as per project convention (grid_size=10000)
        # academic standard is [0, 1]
        pos = torch.tensor(instance, dtype=torch.float32) * 10000.0
        instances.append(pos)
    
    # Standard datasets have fixed size for all instances
    print("Stacking...")
    final_data = torch.stack(instances)
    
    print(f"Saving to {output_path}...")
    torch.save(final_data, output_path)
    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Input .pkl file")
    parser.add_argument("--output", type=str, required=True, help="Output .fast.pt file")
    args = parser.parse_args()
    
    convert_pkl_to_pt(args.input, args.output)
