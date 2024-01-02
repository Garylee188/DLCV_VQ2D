import numpy as np
import matplotlib.pyplot as plt
import json

def sliding_window_find_peak(json_file, window_size=5, stride=1, show_result=False):
    
    with open(json_file, 'r') as file:
        json_file = json.load(file)
    
    x, y = [], []
    for key, val in json_file.items():
        for frame, pts in val.items():
            x.append(int(frame))
            y.append(len(pts))

    # Sliding window to find peak positions
    peak_positions = []
    peak_values = []
    for i in range(0, len(y) - window_size + 1, stride):
        window = y[i:i+window_size]
        
        if np.argmax(window) == window_size // 2:
            peak_position = x[i + window_size // 2]
            peak_positions.append(peak_position)
            peak_values.append(y[i:i+window_size][np.argmax(window)])

    # Select top-k
    k = 3
    topk_ids = np.argsort(peak_values)[-k:]
    topk_peak = np.array([peak_values[ids] for ids in topk_ids])
    topk_frame = np.array([peak_positions[ids] for ids in topk_ids])
    peak_frame = np.max(topk_frame[topk_peak > int(np.mean(topk_peak))])

    # Record all kpts in the peak frame
    for key, val in json_file.items():
        peak_kpts = val[f'{peak_frame}']

    if show_result:
        print('The peak frame: ', peak_frame)
        plt.plot(x, y, label='Curve')
        plt.scatter(peak_frame, y[peak_frame], color='red', label='Peak')
        plt.legend()
        plt.show()
    
    return {
        'peak_frame': peak_frame,
        'peak_kpts': peak_kpts
    }

if __name__ == '__main__':
    test_json_file = './data/train_2.json'
    result = sliding_window_find_peak(test_json_file)
    print(result['peak_frame'])
    print(result['peak_kpts'])