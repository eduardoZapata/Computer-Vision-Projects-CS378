#!/usr/bin/env bash

rm -f test_data/debug_frames/debug_video.mp4
ffmpeg -framerate 15 -i test_data/debug_frames/%d.png -c:v libx264 -r 30 -pix_fmt yuv420p test_data/debug_frames/debug_video.mp4

