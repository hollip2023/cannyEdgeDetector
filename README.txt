https://en.wikipedia.org/wiki/Canny_edge_detector
canny edge detection on RGB24 file, grayspace color only
make:
	gcc main.c -lm
usage:
	./a.out image.raw 640 360
generate RGB24 image:
	ffmpeg -i sample.png -f rawvideo -pix_fmt rgb24 image.raw
display output image:
	ffplay -f rawvideo -video_size 640x360 -pixel_format gray -i canny.raw

