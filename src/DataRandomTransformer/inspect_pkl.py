import pickle
import os

pkl_path = 'train_interleaved_traces_ortho_haar_ident_C_multi_cut.pkl'

with open(pkl_path, 'rb') as f:
    first_bytes = f.read(32)

print('First 32 bytes (hex):', first_bytes.hex())
print('First 32 bytes (ascii, errors replaced):', first_bytes.decode(errors='replace'))

with open(pkl_path, 'rb') as f:
    import pickle
    try:
        data = pickle.load(f)
        print(f"Type: {type(data)}")
        if isinstance(data, dict):
            print(f"Keys: {list(data.keys())}")
            if 'multi_sys_ys' in data:
                arr = data['multi_sys_ys']
                print(f"multi_sys_ys shape: {arr.shape}, dtype: {arr.dtype}")
            for k in list(data.keys())[:3]:
                print(f"Key: {k}, Type: {type(data[k])}")
                try:
                    print(f"Sample: {str(data[k])[:500]}")
                except Exception as e:
                    print(f"Could not print sample for key {k}: {e}")
        elif isinstance(data, list):
            print(f"Length: {len(data)}")
            print(f"First item type: {type(data[0])}")
            print(f"First item sample: {str(data[0])[:500]}")
        else:
            print(f"Data sample: {str(data)[:1000]}")
    except Exception as e:
        print(f"Could not unpickle file: {e}") 