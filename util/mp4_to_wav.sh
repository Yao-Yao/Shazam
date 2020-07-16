id=${1%.mp4}
ffmpeg -i $id.mp4 -ac 1 -y $id.wav
