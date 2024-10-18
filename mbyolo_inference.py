import argparse
from ultralytics import YOLO
import cv2
import os

def process_video(model_path, video_path, output_path=None):
    # Load the trained model
    model = YOLO(model_path)

    # Get model name from the path
    model_name = model_path.split('/')[-3]

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Get video name from the path
    video_name = os.path.splitext(os.path.basename(video_path))[0]

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Define the output video name
    if output_path is None:
        output_path = f"{model_name}_{video_name}_output.mp4"
    
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Process the video
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Run inference on the frame
        results = model(frame)

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # Write the frame to the output video
        out.write(annotated_frame)

    # Release everything when done
    cap.release()
    out.release()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run MAMBA-YOLO inference on a video")
    parser.add_argument("--model", required=True, help="Path to the MAMBA-YOLO model")
    parser.add_argument("--video", required=True, help="Path to the input video")
    parser.add_argument("--output", help="Path to the output video (optional)")
    args = parser.parse_args()

    process_video(args.model, args.video, args.output)
