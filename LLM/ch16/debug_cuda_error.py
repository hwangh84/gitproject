import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import sys

def debug_main():
    print("1. Loading Data...")
    try:
        url = "https://drive.google.com/uc?id=1KOKgZ4qCg49bgj1QNTwk1Vd29soeB27o"
        df = pd.read_csv(url)
        print(f"   Data loaded. Shape: {df.shape}")
    except Exception as e:
        print(f"   Failed to load data: {e}")
        return

    print("2. Preprocessing Data...")
    # Logic from notebook
    y = np.array([1 if value >= 6 else 0 for value in df.rating])
    x = df['review']
    
    # Check labels
    unique_labels = np.unique(y)
    print(f"   Unique labels found: {unique_labels}")
    if not np.all(np.isin(unique_labels, [0, 1])):
        print("   ERROR: Labels contain values other than 0 and 1!")
    
    X_, x_test, y_, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)
    x_train, x_val, y_train , y_val = train_test_split(X_, y_, test_size=0.2, random_state=42, stratify=y_)
    
    print(f"   Train shape: {x_train.shape}, Val shape: {x_val.shape}")

    print("3. Tokenization & Full Data Scan...")
    model_name = "bert-base-multilingual-cased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    vocab_size = tokenizer.vocab_size
    print(f"   Tokenizer vocab size: {vocab_size}")

    # Check for NaNs in text
    if x.isnull().any():
        print("   CRITICAL: Found NaN values in input text!")
        print(x[x.isnull()])
        x = x.fillna("") # Handle for now to proceed
    
    # Tokenize all training data
    print("   Tokenizing all training data (this might take a moment)...")
    try:
        train_encodings = tokenizer(x_train.tolist(), padding=True, truncation=True, return_tensors='pt')
    except Exception as e:
        print(f"   Tokenization failed: {e}")
        return

    input_ids = train_encodings['input_ids']
    print(f"   Input IDs shape: {input_ids.shape}")
    
    # Check max token ID
    max_id = input_ids.max().item()
    min_id = input_ids.min().item()
    print(f"   Max Token ID: {max_id}")
    print(f"   Min Token ID: {min_id}")
    
    if max_id >= vocab_size:
        print(f"   CRITICAL ERROR: Found token ID {max_id} >= vocab size {vocab_size}")
        # Find which index
        bad_indices = (input_ids >= vocab_size).nonzero(as_tuple=True)
        print(f"   Bad indices (first 5): {bad_indices[0][:5]}")
    
    if min_id < 0:
        print(f"   CRITICAL ERROR: Found negative token ID {min_id}")

    print("4. Model Loading...")
    num_labels = 2
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    model_vocab_size = model.config.vocab_size
    print(f"   Model vocab size: {model_vocab_size}")
    
    if max_id >= model_vocab_size:
         print(f"   CRITICAL ERROR: Input token ID {max_id} is larger than model vocabulary {model_vocab_size}!")

    print("5. Running Forward Pass on GPU with full batch (or large chunks)...")
    if torch.cuda.is_available():
        device = torch.device("cuda")
        model.to(device)
        
        # Run in chunks to avoid OOM, but we want to trigger the assert
        batch_size = 32
        dataset_size = len(y_train)
        print(f"   Running inference on {dataset_size} samples...")
        
        import os
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
        model.train()
        
        print(f"   Running training step (forward + backward) on {dataset_size} samples...")
        
        for i in range(0, dataset_size, batch_size):
            optimizer.zero_grad()
            batch_ids = input_ids[i:i+batch_size].to(device)
            batch_mask = train_encodings['attention_mask'][i:i+batch_size].to(device)
            batch_labels = torch.tensor(y_train[i:i+batch_size]).to(device)
            
            try:
                outputs = model(input_ids=batch_ids, attention_mask=batch_mask, labels=batch_labels)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                
                if i % 1000 == 0:
                    print(f"   Batch {i//batch_size} processed. Loss: {loss.item()}")
                    
            except RuntimeError as e:
                print(f"   CRITICAL: Crash at batch {i//batch_size} (indices {i} to {i+batch_size}) during training step")
                print(f"   Error: {e}")
                print(f"   Batch Max ID: {batch_ids.max().item()}")
                print(f"   Batch Labels: {batch_labels.cpu().numpy()}")
                break
        print("   Finished scanning all batches with backward pass.")
    else:
        print("   CUDA not available, skipping GPU scan.")

if __name__ == "__main__":
    debug_main()
