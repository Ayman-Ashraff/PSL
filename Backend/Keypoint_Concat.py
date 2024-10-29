import math
import numpy as np
import os
import multiprocessing


# smaller function to sort files before concatenation
def sort_files(frames):
    finalFrames = []
    for i in range(len(frames)):
        finalFrames.append(int(frames[i].replace('.jpg.npy', '')))
    finalFrames.sort()

    for j in range(len(finalFrames)):
        finalFrames[j] = str(finalFrames[j]) + '.jpg.npy'
    return finalFrames


# function to read npy files of all frames and to concatenate and store them as a 3D array
def concat(actions, name):
    data_path = r"/Users/kayan/Downloads/NewKeyPoints"  # Update to your keypoints directory
    npy_path = f"/Users/kayan/Downloads/60_classes_sorted_concat/{name}"  # Update to your desired output directory
    video_list = []
    i = 1
    for action in actions:
        videos = os.listdir(os.path.join(data_path, action))
        for video in videos:
            augs = os.listdir(os.path.join(data_path, action, video))
            for aug in augs:
                img = []
                frames = os.listdir(os.path.join(data_path, action, video, aug))  # where's the sort
                frames = sort_files(frames)

                for frame in frames:
                    res = np.load(os.path.join(data_path, action, video, aug, frame))
                    img.append(res)
                video_list.append(img)
        print(os.getpid(), ' completed action: ', action)
        print('iteration: ', i)
        i += 1
    X = np.array(video_list)
    np.save(npy_path, X)


# function to generate and store labels for each video and all its augmentations
def labels():
    labels = []
    label_map = {label: num for num, label in enumerate(actions)}
    for action in actions:
        videos = os.listdir(os.path.join(os.getcwd(), action))
        for video in videos:
            augs = os.listdir(os.path.join(os.getcwd(), action, video))
            for aug in augs:
                labels.append(label_map[action])
    Y = np.array(labels)
    np.save(r"/Users/kayan/Downloads/all_labels.npy", Y)  # Update to your desired output path


# function to read and join smaller npy files into one big npy file
def merge(dir_path, save_name):
    arrays = []
    for filename in os.listdir(dir_path):
        if filename.endswith('.npy'):
            file_path = os.path.join(dir_path, filename)
            arr = np.load(file_path)
            arrays.append(arr)

    concatenated_array = np.concatenate(arrays, axis=0)

    save_path = f'/Users/kayan/Downloads/{save_name}.npy'  # Update to your desired output path
    np.save(save_path, concatenated_array)


if __name__ == '__main__':
    DATA_PATH = r"/Users/kayan/Downloads/NewKeyPoints"  # Update to your keypoints directory
    actions = os.listdir(DATA_PATH)
    actions = actions[:60]
    length = math.floor(len(actions) / 6)
    first = actions[:length]
    second = actions[length:length * 2]
    third = actions[length * 2:length * 3]
    fourth = actions[length * 3:length * 4]
    fifth = actions[length * 4:length * 5]
    sixth = actions[length * 5:]

    p1 = multiprocessing.Process(target=concat, args=(first, '1',))
    p2 = multiprocessing.Process(target=concat, args=(second, '2',))
    p3 = multiprocessing.Process(target=concat, args=(third, '3',))
    p4 = multiprocessing.Process(target=concat, args=(fourth, '4',))
    p5 = multiprocessing.Process(target=concat, args=(fifth, '5',))
    p6 = multiprocessing.Process(target=concat, args=(sixth, '6',))

    p1.start()
    p2.start()
    p3.start()
    p4.start()
    p5.start()
    p6.start()

    # Uncomment to run after all arrays have been concatenated
    # merge(r'/Users/kayan/Downloads/Final_Concatenation', 'allarrays')
    # labels()
