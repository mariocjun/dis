import numpy as np
import struct
import os

def generate_large():
    # Model: 100x100 image -> 10,000 columns
    # Signal: 5000 detections -> 5,000 rows
    ROWS = 5000
    COLS = 10000
    
    print(f"Generating Large Matrix {ROWS}x{COLS}...")
    
    # 1. Create Random Matrix (Float64)
    # Using float32 to save generation time, then casting to float64 for writing? 
    # No, C++ expects double.
    # Generate in chunks to save RAM?
    # 50M doubles = 400MB. Easy for modern RAM.
    
    H = np.random.rand(ROWS, COLS).astype(np.float64)
    
    # Sparsify it a bit to be realistic? 
    # Ultrasound matrices are somewhat sparse, but "Dense" solver handles them as dense.
    # Let's keep it random dense to burn CPU.
    
    # Save as Binary for C++ (Column Major)
    # Header: int64 rows, int64 cols
    bin_path = os.path.join("data", "H-large.csv.dense.bin")
    
    print(f"Writing binary to {bin_path}...")
    with open(bin_path, "wb") as f:
        f.write(struct.pack("qq", ROWS, COLS)) # long long (64-bit)
        # Flatten in Fortran order (Column Major)
        f.write(H.flatten(order='F').tobytes())
        
    # Create Dummy CSV to satisfy file checkers if any
    csv_path = os.path.join("data", "H-large.csv")
    with open(csv_path, "w") as f:
        f.write("Dummy CSV for H-large. See .dense.bin")
        
    # Save as NPY for Python
    npy_path = os.path.join("data", "H-large.csv.npy")
    print(f"Writing NPY to {npy_path}...")
    np.save(npy_path, H)
    
    # 2. Generate Signal (g)
    # g = H * x + noise
    x_true = np.zeros(COLS)
    # Set a few pixels
    x_true[5000] = 1.0
    x_true[5050] = 0.5
    
    g = H @ x_true
    
    g_path = os.path.join("data", "g-large.csv")
    print(f"Writing signal to {g_path}...")
    np.savetxt(g_path, g, fmt='%.6f')
    
    print("Done!")

if __name__ == "__main__":
    generate_large()
