import imageio
import os

def create_video(image_folder, video_name):
    images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
    images.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))  # Sort the images by frame number

    frame = imageio.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    video = imageio.get_writer(video_name, fps=5)

    for image in images:
        video.append_data(imageio.imread(os.path.join(image_folder, image)))

    video.close()

num_episodes = 50

# Create a video for the first episode
create_video("recordings/coor_episode_1", "coor_episode_1.mp4")

# Create a video for the last episode
create_video(f"recordings/coor_episode_{num_episodes - 1}", f"coor_episode_{num_episodes - 1}.mp4")