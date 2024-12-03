import cv2  

def get_first_frame(video_path, output_image_path, target_size=(1920, 1080)):  
    """  
    Extract the first frame from a video, resize it to the target size, and save it as an image.  

    Args:  
        video_path (str): Path to the input video file.  
        output_image_path (str): Path to save the resized first frame as an image.  
        target_size (tuple): Target resolution (width, height) to resize the frame.  
    """  
    # Open the video file  
    cap = cv2.VideoCapture(video_path)  

    # Check if the video file is opened successfully  
    if not cap.isOpened():  
        print(f"Error: Unable to open video file {video_path}")  
        return  

    # Read the first frame  
    ret, frame = cap.read()  
    if not ret:  
        print("Error: Unable to read the first frame from the video.")  
        cap.release()  
        return  

    # Resize the frame to the target size  
    resized_frame = cv2.resize(frame, target_size)  

    # Save the resized frame as an image  
    cv2.imwrite(output_image_path, resized_frame)  
    print(f"First frame saved to {output_image_path}")  

    # Release the video capture object  
    cap.release()  


if __name__ == "__main__":  
    # Input video path  
    video_path = "data/video-20241129-170159.mp4"  

    # Output image path  
    output_image_path = "data/first_frame_resized.jpg"  

    # Target resolution  
    target_size = (1920, 1080)  

    # Extract and save the first frame  
    get_first_frame(video_path, output_image_path, target_size)