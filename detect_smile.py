#!/usr/bin/env python3
import argparse
import os
import cv2
from smile_detector import SmileDetector

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Detect smiles in a video file')
    parser.add_argument('input_video', help='Path to the input video file')
    parser.add_argument('--output', '-o', help='Path to save the output video (optional)')
    parser.add_argument('--display', '-d', action='store_true', help='Display the video while processing')
    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.exists(args.input_video):
        print(f"Error: Input file '{args.input_video}' does not exist")
        return 1
    
    # Set default output path if not provided
    if not args.output:
        base_name = os.path.splitext(os.path.basename(args.input_video))[0]
        args.output = f"{base_name}_processed.mp4"
    
    # Initialize the smile detector
    detector = SmileDetector()
    
    # Define callback function for displaying frames
    def display_callback(frame_number, total_frames, frame, results):
        if args.display:
            # Resize frame for display if it's too large
            height, width = frame.shape[:2]
            max_display_width = 1280
            if width > max_display_width:
                scale = max_display_width / width
                frame = cv2.resize(frame, (int(width * scale), int(height * scale)))
            
            # Display the frame
            cv2.imshow('Smile Detection', frame)
            
            # Calculate progress
            progress = int((frame_number / total_frames) * 100)
            print(f"\rProcessing: {progress}% complete", end='')
            
            # Exit if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\nProcessing interrupted by user")
                cv2.destroyAllWindows()
                return False
        else:
            # Print progress without display
            if frame_number % 10 == 0:  # Update every 10 frames
                progress = int((frame_number / total_frames) * 100)
                print(f"\rProcessing: {progress}% complete", end='')
        
        return True
    
    print(f"Processing video: {args.input_video}")
    print(f"Output will be saved to: {args.output}")
    
    # Process the video
    try:
        detector.process_video(args.input_video, args.output, display_callback)
        print("\nProcessing complete!")
        print(f"Output saved to: {args.output}")
    except Exception as e:
        print(f"\nError processing video: {str(e)}")
        return 1
    
    # Clean up
    if args.display:
        cv2.destroyAllWindows()
    
    return 0

if __name__ == "__main__":
    exit(main()) 