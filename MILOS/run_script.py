import os
import numpy as np
#!/usr/bin/env bash
# declare an array variable
# bigger runs (for night)
#K_array=( 512 256 128 64 32 16 8 4 2 )
#           9  8    7   6  5  4 3 2 1
#D_array=( 1 2 3 4 5 6 7 8 9 10 11 12)

# smaller runs (for day) 
#K_array=( 64 32 16 8 4 2)
#         6  5  4  3 2 1
#D_array=( 6 5 4 3 2 1 )

# countinous vq-vae learns with K=512 D=256 and M=2
D_array = np.array([256])  #np.array([256, 128, 64, 32])
M_array = np.array([1]) #np.array([0, 1, 3])   #np.array([1, 2, 3, 4, 5, 6])
K_array = 2 ** np.arange(1, 12)    # from 1 bit to 19bits (ground truth is 14bits)

run_id=0
input_bits=14
# to avoid this error:
# Error: mkl-service + Intel(R) MKL: MKL_THREADING_LAYER=INTEL is incompatible with libgomp.so.1 library.
#         Try to import numpy first or set the threading layer accordingly. Set MKL_SERVICE_FORCE_INTEL to force it.
# taken from https://github.com/pytorch/pytorch/issues/37377
os.environ['MKL_THREADING_LAYER'] = 'GNU'

# it is beginning to learn for the following combination of (K,D,M)
# and number of bits is log2(K)*M^2:
# 2) Running for K = 1024 & D = 256 & M = 1 (i.e. bits = 10):
# 7) Running for K = 32768 & D = 256 & M = 1 (i.e. bits = 15):
# 11) Running for K = 524288 & D = 256 & M = 1 (i.e. bits = 19):



# CORRECT: # it is beginning to learn for the following combination of (K,D,M)
# 6) Running for K = 16384 & D = 256 & M = 1 (i.e. bits = 14):
# 7) Running for K = 128 & D = 256 & M = 0 (i.e. bits = 7):
for d in D_array:
    for m in M_array:
        for k in K_array:
            compressed_number_of_bits_per_image = int(np.ceil((m+1)**2 * np.log2(k)))
            
            if compressed_number_of_bits_per_image > 50:
                print(f"Pass the run for K = {k} & D = {d} & M = {m} (i.e. bits = {compressed_number_of_bits_per_image}):\n")
                continue
            
            run_id += 1
            compression_gain = round(input_bits / compressed_number_of_bits_per_image,3)
            print(f"{run_id}) Running for K = {k} & D = {d} & M = {m} (i.e. bits = {compressed_number_of_bits_per_image}):\n")
            command = f"python /home/novakovm/iris/MILOS/autoencoders.py {k} {d} {run_id} {m}"
            os.system(command)
            print("\n****************************************\n")