import argparse
import random
from datasets import load_dataset
import os

def read_fasta(fasta_file):
    sequences = []
    with open(fasta_file, 'r') as file:
        sequence = ""
        for line in file:
            if line.startswith(">"):
                if sequence:
                    sequences.append(sequence)
                    sequence = ""
            else:
                sequence += line.strip()
        if sequence:
            sequences.append(sequence)
    return sequences

def preprocess_protgpt2(sequences):
    def split_sequence(seq, line_length=60):
        return '\n'.join(seq[i:i+line_length] for i in range(0, len(seq), line_length))
    return ["<|endoftext|>\n" + split_sequence(seq) for seq in sequences]

def preprocess_progen(sequences):
    return sequences

def preprocess_rita(sequences):
    return sequences  

PREPROCESSORS = {
    "protgpt2": preprocess_protgpt2,
    "progen": preprocess_progen,
    "rita": preprocess_rita,
}

def split_dataset(sequences):
    random.shuffle(sequences)
    n = len(sequences)
    train = sequences[:int(0.9*n)]
    valid = sequences[int(0.9*n):int(0.95*n)]
    test = sequences[int(0.95*n):]
    return train, valid, test

def save_splits(train, valid, test, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    with open(f"{out_dir}/train.txt", 'w') as f: f.write("\n".join(train))
    with open(f"{out_dir}/validation.txt", 'w') as f: f.write("\n".join(valid))
    with open(f"{out_dir}/test.txt", 'w') as f: f.write("\n".join(test))

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--fasta_file", required=True)
#     parser.add_argument("--model", required=True, choices=PREPROCESSORS.keys())
#     parser.add_argument("--output_dir", default="./splits")
#     args = parser.parse_args()

#     raw_sequences = read_fasta(args.fasta_file)
#     preprocessor = PREPROCESSORS[args.model]
#     processed_sequences = preprocessor(raw_sequences)
    
#     train, valid, test = split_dataset(processed_sequences)
#     save_splits(train, valid, test, args.output_dir)

#     dataset = load_dataset("text", data_files={
#         "train": f"{args.output_dir}/train.txt",
#         "validation": f"{args.output_dir}/validation.txt",
#         "test": f"{args.output_dir}/test.txt"
#     })

#     print("[INFO] Dataset loaded successfully:", dataset)
