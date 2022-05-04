from pymediainfo import MediaInfo
from PIL import Image
import subprocess
from torchvision.transforms.functional import resize

VIDEO_EXTENSION = ['rm', 'webm', 'mpg', 'mp2', 'mpeg', 'mpe', 'asf',
                   'mpv', 'ogg', 'mp4', 'm4p', 'm4v', 'avi',
                   'wmv', 'mov', 'qt', 'flv', 'swf', 'avchd']


def parse_video(video):
    meta = dict()
    meta['path'] = video
    code = True
    try:
        media_info = MediaInfo.parse(video)
        for track in media_info.tracks:
            if track.track_type == 'General':
                meta['file_name'] = track.file_name + '.' + track.file_extension
                meta['file_extension'] = track.file_extension
                meta['format'] = track.format
                meta['duration'] = track.duration
                meta['frame_count'] = track.frame_count
                meta['frame_rate'] = track.frame_rate
            elif track.track_type == 'Video':
                meta['width'] = int(track.width)
                meta['height'] = int(track.height)
                meta['rotation'] = float(track.rotation) if track.rotation is not None else 0.
                meta['codec'] = track.codec
    except Exception as e:
        code = False

    return code, meta

def decode_video(video, frame_dir, decode_rate):
    cmd = ['ffmpeg',
           '-i', video,
           '-vsync', '2',
           '-map', '0:v:0',
           '-q:v', '0',
           '-vf', f'fps={decode_rate}',
           '-f', 'image2',
           f'{frame_dir}/%6d.jpg']

    p = subprocess.Popen(args=cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=False)
    import pdb; pdb.set_trace()
    _ = p.communicate()
    code = True if p.returncode != 1 else False
    return code

def decode_video2(video, frame_dir, decode_rate):
    cmd = ['ffmpeg',
           '-ss', '00:00:0'
           '-i', f'{video}',
           '-r', f'{decode_rate}',
           '-f', 'image2',
           f'{frame_dir}/%6d.jpg']

    p = subprocess.Popen(args=cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=False)
    import pdb; pdb.set_trace()
    _ = p.communicate()
    code = True if p.returncode != 1 else False
    return code


def decode_video_to_pipe(video, meta, decode_rate, size):
    frames = []
    w, h = (meta['width'], meta['height']) if meta['rotation'] not in [90, 270] else (meta['height'], meta['width'])
    command = ['ffmpeg',
               '-hide_banner', '-loglevel', 'panic',
               '-vsync', '2',
               '-i', video,
               '-pix_fmt', 'bgr24',  # color space
               '-vf', f'fps={decode_rate}',  # '-r', str(decode_rate),
               '-q:v', '0',
               '-vcodec', 'rawvideo',  # origin video
               '-f', 'image2pipe',  # output format : image to pipe
               'pipe:1']
    pipe = subprocess.Popen(command, stdout=subprocess.PIPE, bufsize=w * h * 3)
    while True:
        raw_image = pipe.stdout.read(w * h * 3)
        pipe.stdout.flush()
        try:
            image = Image.frombuffer('RGB', (w, h), raw_image, "raw", 'BGR', 0, 1)
        except ValueError as e:
            break

        if size:
            image = resize(image, size)
        frames.append(image)
    return frames
