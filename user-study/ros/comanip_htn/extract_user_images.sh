# Arg1: data folder | Arg2: user name | Arg3: mode

#mkdir $1/user_$2/dump_$2_$3
#ffmpeg -i $1/user_$2/User-$2_mode-$3_video.avi -vf fps=5 $1/user_$2/dump_$2_$3/$2_$3-%d.png

mkdir $1/user_$2/dump_$2_adaptive
ffmpeg -i $1/user_$2/User-$2_mode-adaptive_video.avi -vf fps=5 $1/user_$2/dump_$2_adaptive/$2_adaptive-%d.png

mkdir $1/user_$2/dump_$2_fixed
ffmpeg -i $1/user_$2/User-$2_mode-fixed_video.avi -vf fps=5 $1/user_$2/dump_$2_fixed/$2_fixed-%d.png
