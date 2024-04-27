import os
from os.path import join
import numpy as np
import sklearn.decomposition as decomp
from sklearn.mixture import GaussianMixture as GMM
from sklearn.svm import LinearSVC
from video_representation import VideoRepresentation
from transforms import *
from settings import *
from visualize import *
import subprocess
import numpy as np
import tensorflow as tf
import cv2
import time


class DetectorAPI:
    def __init__(self, path_to_ckpt):
        self.path_to_ckpt = path_to_ckpt

        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.compat.v1.GraphDef()  # Changed here
            with tf.io.gfile.GFile(self.path_to_ckpt, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        self.default_graph = self.detection_graph.as_default()
        self.sess = tf.compat.v1.Session(graph=self.detection_graph, config=tf.compat.v1.ConfigProto())

        # Definite input and output Tensors for detection_graph
        self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
        # Each box represents a part of the image where a particular object was detected.
        self.detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        self.detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        self.detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        self.num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')

    def processFrame(self, image):
        # Expand dimensions since the trained_model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image, axis=0)
        # Actual detection.
        start_time = time.time()
        (boxes, scores, classes, num) = self.sess.run(
            [self.detection_boxes, self.detection_scores, self.detection_classes, self.num_detections],
            feed_dict={self.image_tensor: image_np_expanded})
        end_time = time.time()

        # print("Elapsed Time:", end_time-start_time)

        im_height, im_width,_ = image.shape
        boxes_list = [None for i in range(boxes.shape[1])]
        for i in range(boxes.shape[1]):
            boxes_list[i] = (int(boxes[0,i,0] * im_height),
                        int(boxes[0,i,1]*im_width),
                        int(boxes[0,i,2] * im_height),
                        int(boxes[0,i,3]*im_width))

        return boxes_list, scores[0].tolist(), [int(x) for x in classes[0].tolist()], int(num[0])

    def close(self):
        self.sess.close()
        self.default_graph.close()


def write_2d_array_to_file(array, filename):
    # Open the file in 'w' mode, which creates the file if it doesn't exist
    with open(filename, 'w') as file:
        for row in array:
            # Convert each element in the row to a string and join them with a space
            row_string = ' '.join(map(str, row))
            # Write the row to the file followed by a newline character
            file.write(row_string + '\n')

bounding_boxes=[]
def bounding_box_calculator(path,BB_path):
    model_path = r'C:\Users\shiva\Downloads\dense-trajectories-action-recognition-20240312T070544Z-001\dense-trajectories-action-recognition\frozen_inference_graph.pb'
    odapi = DetectorAPI(path_to_ckpt=model_path)
    threshold = 0.7
    cap = cv2.VideoCapture(path)
    frame_no=0
    while True:
        tmp_container=[]
        r, img = cap.read()
        if(r==False):
            break
        original_height, original_width = img.shape[:2]
        img = cv2.resize(img, (1280, 720))
        resized_height, resized_width = img.shape[:2]

        # Calculate the scaling factors
        scale_x = original_width / resized_width
        scale_y = original_height / resized_height
        

        if(frame_no==0):
            print(img.shape)
        tmp_container.append(frame_no)
       
        boxes, scores, classes, num = odapi.processFrame(img)

        # Visualization of the results of a detection.

        for i in range(len(boxes)):
            # Class 1 represents human
            if classes[i] == 1 and scores[i] > threshold:
                # print("Confidence  - ",scores[i])
                box = boxes[i]
                tmp_container.append(box[1]*scale_x)
                tmp_container.append(box[0]*scale_y)
                tmp_container.append(box[3]*scale_x)
                tmp_container.append(box[2]*scale_y)
                tmp_container.append(scores[i])
                # print("Frame number ",frame_no)
                # print(box)
                cv2.rectangle(img,(box[1],box[0]),(box[3],box[2]),(255,0,0),2)
                # Draw circles at the corners
                cv2.circle(img, (box[1], box[0]), 5, (0, 255, 0), -1)  # Top-left corner
                cv2.circle(img, (box[3], box[0]), 5, (0, 255, 0), -1)  # Top-right corner
                cv2.circle(img, (box[1], box[2]), 5, (0, 255, 0), -1)  # Bottom-left corner
                cv2.circle(img, (box[3], box[2]), 5, (0, 255, 0), -1)  # Bottom-right corner

                # Convert coordinates to strings
                top_left_str = f"({box[1]}, {box[0]})"
                top_right_str = f"({box[3]}, {box[0]})"
                bottom_left_str = f"({box[1]}, {box[2]})"
                bottom_right_str = f"({box[3]}, {box[2]})"

                # Display the coordinates
                cv2.putText(img, top_left_str, (box[1] - 50, box[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(img, top_right_str, (box[3] + 10, box[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(img, bottom_left_str, (box[1] - 50, box[2] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(img, bottom_right_str, (box[3] + 10, box[2] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)


        bounding_boxes.append(tmp_container)
        frame_no+=1
        cv2.imshow("preview", img)
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break
    write_2d_array_to_file(bounding_boxes,BB_path)

def call_exe_with_arguments(exe_path, arguments):
    try:
        # Run the .exe file with the provided arguments
        subprocess.run([exe_path] + arguments, check=True)
        print("Execution successful.")
    except subprocess.CalledProcessError as e:
        # Handle any errors
        print("Error:", e)


def convert_to_float16(arr):
    return arr.astype(np.float16)

def main(already_computed_descriptors=False):
    try:
        os.mkdirs(video_descriptors_path)
    except:
        pass

    train_videos = []
    test_videos = []

    # COMPUTE DESCRIPTORS
    if not already_computed_descriptors:
        for directory in next(os.walk(data_dir))[1]:
            directory_path = join(data_dir, directory)
            print(f'\n________EXTRACTING DESCRIPTORS FROM {directory_path}')
            for filename in os.listdir(directory_path):
                filepath = join(directory_path, filename)
                if '.avi' in filename and os.path.isfile(filepath):
                    tmpfilename = filename
                    # Split the filename into its root and extension
                    root, ext = os.path.splitext(tmpfilename)
                    # Replace the extension with '.txt'
                    new_filename = root + ".txt"
                    BB_path = join('BoundingBoxes', new_filename)
                    bounding_box_calculator(filepath,BB_path)
                    exe_path =r"C:\Users\shiva\source\repos\Project2\x64\Release\Project2.exe"
                    call_exe_with_arguments(exe_path,[filepath,BB_path])
                   

    # TRAIN
    train_lines = []
    cntr=0
    with open(join(data_dir, 'train.txt'), 'r') as train_f:
        train_lines = train_f.readlines()
    for l in train_lines:
        filepath, label = l.split()
        descriptor_path = join(video_descriptors_path,
                               f'{filepath.split("/")[1]+"-descriptors.txt"}')
        # print(descriptor_path)
        video_representation = VideoRepresentation(filepath, np.loadtxt(descriptor_path, dtype=np.float16), label)
        print(cntr)
        cntr+=1
        train_videos.append(video_representation)
    # print([v.descriptors for v in train_videos])
    # for v in train_videos:
    all_train_descriptors = np.concatenate([v.descriptors for v in train_videos], axis=0,dtype=np.float16)
    # all_train_descriptors = np.concatenate([convert_to_float16(v.descriptors) for v in train_videos], axis=0)
    print(f'total number of train descriptors: {all_train_descriptors.shape[0]}')
    print(f'length of each train descriptor: {all_train_descriptors.shape[1]}')

    # init and fit the pca
    pca = decomp.PCA(pca_num_components)
    pca = pca.fit(all_train_descriptors)
    
    # transform descriptors of each video
    for v in train_videos:
        v.pca_descriptors = pca.transform(v.descriptors)

    # concatenate the pca-transformed descriptors, to not transform the whole data one extra time
    all_train_descriptors = np.concatenate([v.pca_descriptors for v in train_videos], axis=0)
    print(f'length each train descriptor after pca: {all_train_descriptors.shape[1]}')

    # learn GMM model
    gmm = GMM(n_components=gmm_n_components, covariance_type='diag')
    gmm.fit(all_train_descriptors)

    # compute fisher vectors for each train video
    for v in train_videos:
        v.fisher_vector = fisher_from_descriptors(v.pca_descriptors, gmm)
    print('calculated Fisher vectors')

    # initialize and fit a linear SVM
    svm = LinearSVC()
    svm.fit(X=[v.fisher_vector for v in train_videos], y=[v.label for v in train_videos])
    print('fitted SVM')

    # TEST
    test_lines = []
    with open(join(data_dir, 'test.txt'), 'r') as test_f:
        test_lines = test_f.readlines()
    for l in test_lines:
        filepath, label = l.split()
        descriptor_path = join(video_descriptors_path,
                               f'{filepath.split("/")[1]+ "-descriptors.txt"}')
        video_representation = VideoRepresentation(filepath, np.loadtxt(descriptor_path), label)
        test_videos.append(video_representation)

    # reduce dimension of all test descriptors using pca fitted on train data
    for v in test_videos:
        v.pca_descriptors = pca.transform(v.descriptors)
    print('reduced dimensions of the test data')

    # calculate a fisher vector for each test video based on the gmm model fit on the train data
    for v in test_videos:
        v.fisher_vector = fisher_from_descriptors(v.pca_descriptors, gmm)
    print('calculated Fisher vectors on the test data')

    # predict the labels of the test videos
    accuracy = svm.score(X=[v.fisher_vector for v in test_videos], y=[v.label for v in test_videos])
    print(f'accuracy: {accuracy}')
    prediction = svm.predict(X=[v.fisher_vector for v in test_videos])
    for i, v in enumerate(test_videos):
        v.predicted_label = prediction[i]
    print('prediction by video: index, true label, predicted label, path\n')
    for i, v in enumerate(test_videos):
        print(f'{i}    gt: {v.label}    pred: {v.predicted_label}   {v.filepath}')


if __name__ == '__main__':
    # to test trajectories on a single video
    # trajectories_from_video('data/UnevenBars/v_UnevenBars_g01_c01.avi', vis_flow=False, vis_trajectories=True)

    main(already_computed_descriptors=True)
   
