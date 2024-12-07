#a script that converts a pickle file to a hdfs and npy file
import numpy as np
import h5py
import pickle
import sys
import os
import gzip

# # Add the parent directory to the Python path
# parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# sys.path.append(parent_dir)


filepath = "outputs/GPT2/241117_204226.922f5f_rotDiagA_gauss_C_lr_1.584893192461114e-05_num_train_sys_40000/prediction_errors_gauss_C_step=474.ckpt/gaussA_err_lss"

# Load the pickle file
with open(filepath + ".pkl", "rb") as f:
    data = pickle.load(f)
    f.close()

print("Data type: ", type(data))
# print the first item in the data
print("pickle:", data["MOP"][0])

#print size in GB of the pickle file
print("Size of the pickle file in GB: ", os.path.getsize(filepath + ".pkl")/(1024**3))

# Save the data to a HDF5 file
with h5py.File(filepath + ".hdf5", "w") as f:
    for key in data.keys():
        f.create_dataset(key, data=data[key], compression="gzip")

#print size in GB of the hdf5 file
print("Size of the hdf5 file in GB: ", os.path.getsize(filepath + ".hdf5")/(1024**3))

#save the np array to a npy file
np.savez_compressed(filepath + ".npz", data)

#print size in GB of the npy file
print("Size of the npy file in GB: ", os.path.getsize(filepath + ".npz")/(1024**3))



#load the hdf5 file and print the first item
with h5py.File(filepath + ".hdf5", "r") as f:
    print("hdf5", "MOP", f["MOP"][0])

#load the npy file and print the first item
npy_data = np.load(filepath + ".npz", allow_pickle=True)
print("type npy data", type(npy_data))
# print("npy:", npy_data["MOP"][0])
# print("npz:", npy_data[0]["obs"])

# Compress and save the pickle file using gzip
with gzip.open(filepath + ".pkl.gz", "wb") as f:
    pickle.dump(data, f)

# Print size in MB of the compressed pickle file
print("Size of the compressed pickle file in MB: ", os.path.getsize(filepath + ".pkl.gz")/(1024*1024))

print("npy data:", npy_data[0])





