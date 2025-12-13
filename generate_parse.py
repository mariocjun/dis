import numpy as np
import struct
import os

def generate_parse():
    # Model: 30x30 image -> 900 columns
    # Signal: 50,000 detections -> 50,000 rows (Heavy Parsing)
    ROWS = 50000
    COLS = 900 # 30x30
    
    print(f"Generating Parse Heavy Matrix {ROWS}x{COLS}...")
    
    H = np.random.rand(ROWS, COLS).astype(np.float64)
    
    # Save as Binary for C++
    bin_path = os.path.join("data", "H-parse.csv.dense.bin")
    print(f"Writing binary to {bin_path}...")
    with open(bin_path, "wb") as f:
        f.write(struct.pack("qq", ROWS, COLS))
        f.write(H.flatten(order='F').tobytes())
        
    csv_path = os.path.join("data", "H-parse.csv")
    with open(csv_path, "w") as f:
        f.write("Dummy")
        
    npy_path = os.path.join("data", "H-parse.csv.npy")
    np.save(npy_path, H)
    
    # Generate Signal (g)
    x_true = np.zeros(COLS)
    x_true[100] = 1.0
    g = H @ x_true
    
    g_path = os.path.join("data", "g-parse.csv")
    print(f"Writing signal to {g_path}...")
    np.savetxt(g_path, g, fmt='%.6f')
    
    print("Done!")

if __name__ == "__main__":
    generate_parse()
