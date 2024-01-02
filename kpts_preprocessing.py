import numpy as np

def generate_random_points(num_points, xmin, xmax, ymin, ymax):
    x_points = np.random.uniform(xmin, xmax, num_points)
    y_points = np.random.uniform(ymin, ymax, num_points)
    points = np.column_stack((x_points, y_points))
    return points

def point_preprocess(kpts_list, max_kpts=128):
    # peak_result = sliding_window_find_peak(json_file=json_file)
    kpts_array = []
    for kpt in kpts_list:
        kpts_array.append([kpt['x'], kpt['y'], kpt['score']])
    kpts_array = np.array(kpts_array)  # shape = [n_kpts, 3(x, y, score)]

    n_kpts = kpts_array.shape[0]
    if n_kpts > max_kpts:
        sorted_indices = np.argsort(kpts_array[:, 2])  # sort from min to max
        res_kpts = kpts_array[sorted_indices]
        res_kpts = res_kpts[(n_kpts-max_kpts):, :2]

    elif n_kpts < max_kpts:
        xmin = np.min(kpts_array[:, 0], axis=0)
        xmax = np.max(kpts_array[:, 0], axis=0)
        ymin = np.min(kpts_array[:, 1], axis=0)
        ymax = np.max(kpts_array[:, 1], axis=0)
        random_pts = generate_random_points(max_kpts-n_kpts, xmin, xmax, ymin, ymax)
        res_kpts = np.row_stack((kpts_array[:, :2], random_pts))

    else:
        res_kpts = kpts_array[:, :2]
    
    return res_kpts

if __name__ == '__main__':
    json_file = './data/train_1.json'
    test_list = [
            {
                "fnum": 74,
                "x": 63.423763275146484,
                "y": 149.68202209472656,
                "score": 0.3820335268974304
            },
            {
                "fnum": 74,
                "x": 88.62067413330078,
                "y": 157.7555694580078,
                "score": 0.47718003392219543
            },
            {
                "fnum": 74,
                "x": 103.92797088623047,
                "y": 158.7281494140625,
                "score": 0.5641703605651855
            },
            {
                "fnum": 74,
                "x": 113.21620178222656,
                "y": 159.78211975097656,
                "score": 0.5211888551712036
            }
        ]
    res_kpts = point_preprocess(test_list, max_kpts=128)
    print(len(res_kpts))