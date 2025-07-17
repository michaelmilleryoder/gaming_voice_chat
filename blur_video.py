import cv2
from tqdm import tqdm



def detect_bounding_box(vid, detector):
    img_H = vid.shape[0]
    img_W = vid.shape[1]
    detections = detector.detect(vid)[1]
    try:
        faces = [ d[0:4].astype(int) for d in detections ] if detections is not None else []
    except Exception as e:
        print(f"Error in detection: {e}")
        print(faces)
    for x, y, w, h in faces:
        cv2.rectangle(vid, (x, y), (x + w, y + h), (0, 255, 0), 4)
    return faces



def blurBoxes(image, boxes):
    """
    Argument:
    image -- the image that will be edited as a matrix
    boxes -- list of boxes that will be blurred each element must be a dictionary that has [id, score, x1, y1, x2, y2] keys

    Returns:
    image -- the blurred image as a matrix
    """
    
    if boxes is None or len(boxes) == 0:
        return image
    for box in boxes:
        # unpack each box

        # x1, y1 = box["x1"], box["y1"]
        # x2, y2 = box["x2"], box["y2"]

        x, y, w, h = box
        x1, y1 = x, y
        x2, y2 = x + w, y + h

        # crop the image due to the current box
        sub = image[y1:y2, x1:x2]
        if sub.shape[0] == 0 or sub.shape[1] == 0:
            continue

        # apply GaussianBlur on cropped area
        blur = cv2.blur(sub, (25, 25))

        # paste blurred image on the original image
        image[y1:y2, x1:x2] = blur

    return image



def blur(input_file_path, output_file_path, tqdm_index):
    if output_file_path is None:
        output_file_path = input_file_path.replace('.mp4', '_blurred.mp4')

    detector = cv2.FaceDetectorYN.create(
        "yunet_n_640_640.onnx", 
        "", 
        (640, 640),
        score_threshold=0.5,

    )

    # open video
    capture = cv2.VideoCapture(input_file_path)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output = cv2.VideoWriter(output_file_path, fourcc,
                                20.0, (int(capture.get(3)), int(capture.get(4))))
    num_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_counter = 0
    for i in tqdm(list(range(num_frames)), position=tqdm_index, desc=f"Processing {input_file_path}"):
        # read frame by frame
        _, frame = capture.read()
        frame_counter += 1

        # the end of the video?
        if frame is None:
            break

        if i == 0:
            detector.setInputSize((frame.shape[1], frame.shape[0]))

        faces = detect_bounding_box(frame, detector)

        # apply blurring
        frame = blurBoxes(frame, faces)

        output.write(frame)

    capture.release()
    output.release()
    with open(os.path.join(os.path.dirname(output_file_path), 'processed_videos.txt'), 'a') as f:
        f.write(f"{os.path.basename(input_file_path)}\n")
    
    



import argparse
import os
from multiprocessing.dummy import Pool as ThreadPool

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Blur faces in a set of videos.")
    parser.add_argument("--directory", type=str, help="Path to the video files.")
    parser.add_argument("--output", type=str, help="Output directory for blurred videos.")
    parser.add_argument("--threads", type=int, default=8, help="Number of threads to use for processing.")
    args = parser.parse_args()

    if not os.path.exists(args.directory):
        raise FileNotFoundError(f"Directory {args.directory} does not exist.")
    
    if args.output is None:
        args.output = args.directory
    
    if not os.path.exists(args.output):
        os.makedirs(args.output)
        # Create a text file to track processed videos
        with open(os.path.join(args.output, 'processed_videos.txt'), 'w') as f:
            f.write("Processed videos:\n")
    video_files = [f for f in os.listdir(args.directory) if f.endswith('.mp4')]
    if len(video_files) == 0:
        raise FileNotFoundError(f"No video files found in {args.directory}.")
    video_files = [os.path.join(args.directory, f) for f in video_files]
    print(f"Found {len(video_files)} video files in {args.directory}.")
    pool = ThreadPool(args.threads)

    pool.starmap(blur, [(video_file, os.path.join(args.output, os.path.basename(video_file)), i) for i, video_file in enumerate(video_files)])

    pool.close()
    pool.join()

