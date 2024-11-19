import os
import argparse
import cv2
from BlurryFaces.DetectorAPI import Detector
from multiprocessing.dummy import Pool as ThreadPool



def blurBoxes(image, boxes):
    """
    Argument:
    image -- the image that will be edited as a matrix
    boxes -- list of boxes that will be blurred each element must be a dictionary that has [id, score, x1, y1, x2, y2] keys

    Returns:
    image -- the blurred image as a matrix
    """

    for box in boxes:
        # unpack each box
        x1, y1 = box["x1"], box["y1"]
        x2, y2 = box["x2"], box["y2"]

        # crop the image due to the current box
        sub = image[y1:y2, x1:x2]

        # apply GaussianBlur on cropped area
        blur = cv2.blur(sub, (25, 25))

        # paste blurred image on the original image
        image[y1:y2, x1:x2] = blur

    return image



def blur(input_file_path, output_file_path=None):  
    if output_file_path is None:
        output_file_path = input_file_path.replace('.mp4', '_blurred.mp4')

    # assignmodel path and threshold
    model_path = "./BlurryFaces/face_model/face.pb"
    threshold = 0.7

    # create detection object
    detector = Detector(model_path=model_path, name="detection")

    # open video
    capture = cv2.VideoCapture(input_file_path)

    # video width = capture.get(3)
    # video height = capture.get(4)
    # video fps = capture.get(5)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output = cv2.VideoWriter(output_file_path, fourcc,
                                20.0, (int(capture.get(3)), int(capture.get(4))))

    frame_counter = 0
    while True:
        # read frame by frame
        _, frame = capture.read()
        frame_counter += 1

        # the end of the video?
        if frame is None:
            break

        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break
        # real face detection
        faces = detector.detect_objects(frame, threshold=threshold)

        # apply blurring
        frame = blurBoxes(frame, faces)

        # show image
        # cv2.imshow('blurred', frame)
    # if image will be saved then save it 
    output.write(frame)
    print('Blurred video has been saved successfully at',
            output_file_path, 'path')

    # when any key has been pressed then close window and stop the program

    cv2.destroyAllWindows()

def blur_directory(directory, num_threads=4):
    files = os.listdir(directory)
    for file in files:
        if file.endswith('.mp4'):
            pool = ThreadPool(num_threads)
            pool.map(blur, [os.path.join(directory, file)])
            
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Blur faces in a video')
    parser.add_argument('--directory', type=str, help='Directory path')
    parser.add_argument('--threads', type=int, help='Number of threads')
    args = parser.parse_args()

    if args.directory:
        blur_directory(args.directory, args.threads)
    else:
        print('Please provide input and output file paths or directory path')