import imageio.v2 as imageio
import glob

# Get a sorted list of all frame files
frame_files = sorted(glob.glob('frames/*.png'))

# Create a writer object
writer = imageio.get_writer('video.mp4', fps=100)

# Iterate through the frame files and add them to the video
for frame_file in frame_files:
    writer.append_data(imageio.imread(frame_file))

# Close the writer to finalize the video
writer.close()