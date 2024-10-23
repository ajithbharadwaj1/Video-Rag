import os
from dotenv import load_dotenv, find_dotenv
from io import StringIO, BytesIO
import textwrap
from typing import Iterator, TextIO, List, Dict, Any, Optional, Sequence, Union
from enum import auto, Enum
import base64
import glob
from tqdm import tqdm
from pytubefix import YouTube, Stream
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.formatters import WebVTTFormatter
import cv2
import json
import PIL
from PIL import Image
import dataclasses
import random
from datasets import load_dataset
import webvtt
from moviepy.editor import VideoFileClip
from os import path as osp
import whisper

def download_video(video_url, path='/tmp/'):
    print(f'Getting video information for {video_url}')
    if not video_url.startswith('http'):
        return os.path.join(path, video_url)

    filepath = glob.glob(os.path.join(path, '*.mp4'))
    if len(filepath) > 0:
        return filepath[0]

def get_transcript_vtt(video_url, path='/tmp'):
    video_id = get_video_id_from_url(video_url)
    filepath = os.path.join(path,'captions.vtt')
    if os.path.exists(filepath):
        return filepath

def get_video_id_from_url(video_url):
    """
    Examples:
    - http://youtu.be/SA2iWivDJiE
    - http://www.youtube.com/watch?v=_oPAwA_Udwc&feature=feedu
    - http://www.youtube.com/embed/SA2iWivDJiE
    - http://www.youtube.com/v/SA2iWivDJiE?version=3&amp;hl=en_US
    """
    import urllib.parse
    url = urllib.parse.urlparse(video_url)
    if url.hostname == 'youtu.be':
        return url.path[1:]
    if url.hostname in ('www.youtube.com', 'youtube.com'):
        if url.path == '/watch':
            p = urllib.parse.parse_qs(url.query)
            return p['v'][0]
        if url.path[:7] == '/embed/':
            return url.path.split('/')[2]
        if url.path[:3] == '/v/':
            return url.path.split('/')[2]

    return video_url

def str2time(strtime):
    # strip character " if exists
    strtime = strtime.strip('"')
    # get hour, minute, second from time string
    hrs, mins, seconds = [float(c) for c in strtime.split(':')]
    # get the corresponding time as total seconds 
    total_seconds = hrs * 60**2 + mins * 60 + seconds
    total_miliseconds = total_seconds * 1000
    return total_miliseconds

def maintain_aspect_ratio_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    # Grab the image size and initialize dimensions
    dim = None
    (h, w) = image.shape[:2]

    # Return original image if no need to resize
    if width is None and height is None:
        return image

    # We are resizing height if width is none
    if width is None:
        # Calculate the ratio of the height and construct the dimensions
        r = height / float(h)
        dim = (int(w * r), height)
    # We are resizing width if height is none
    else:
        # Calculate the ratio of the width and construct the dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # Return the resized image
    return cv2.resize(image, dim, interpolation=inter)

#Use when both Transcript and Image are available
def extract_and_save_frames_and_metadata(
        path_to_video, 
        path_to_transcript, 
        path_to_save_extracted_frames,
        path_to_save_metadatas):
    
    # metadatas will store the metadata of all extracted frames
    metadatas = []

    # load video using cv2
    video = cv2.VideoCapture(path_to_video)
    # load transcript using webvtt
    trans = webvtt.read(path_to_transcript)
    
    # iterate transcript file
    # for each video segment specified in the transcript file
    for idx, transcript in enumerate(trans):
        # get the start time and end time in seconds
        start_time_ms = str2time(transcript.start)
        end_time_ms = str2time(transcript.end)
        # get the time in ms exactly 
        # in the middle of start time and end time
        mid_time_ms = (end_time_ms + start_time_ms) / 2
        # get the transcript, remove the next-line symbol
        text = transcript.text.replace("\n", ' ')
        # get frame at the middle time
        video.set(cv2.CAP_PROP_POS_MSEC, mid_time_ms)
        success, frame = video.read()
        if success:
            # if the frame is extracted successfully, resize it
            image = maintain_aspect_ratio_resize(frame, height=350)
            # save frame as JPEG file
            img_fname = f'frame_{idx}.jpg'
            img_fpath = osp.join(
                path_to_save_extracted_frames, img_fname
            )
            cv2.imwrite(img_fpath, image)

            # prepare the metadata
            metadata = {
                'extracted_frame_path': img_fpath,
                'transcript': text,
                'video_segment_id': idx,
                'video_path': path_to_video,
                'mid_time_ms': mid_time_ms,
            }
            metadatas.append(metadata)

        else:
            print(f"ERROR! Cannot extract frame: idx = {idx}")

    # save metadata of all extracted frames
    fn = osp.join(path_to_save_metadatas, 'metadatas.json')
    with open(fn, 'w') as outfile:
        json.dump(metadatas, outfile)
    return metadatas


def _processText(text: str, maxLineWidth=None):
    if (maxLineWidth is None or maxLineWidth < 0):
        return text

    lines = textwrap.wrap(text, width=maxLineWidth, tabsize=4)
    return '\n'.join(lines)

# helper function for convert time in second to time format for .vtt or .srt file
def format_timestamp(seconds: float, always_include_hours: bool = False, fractionalSeperator: str = '.'):
    assert seconds >= 0, "non-negative timestamp expected"
    milliseconds = round(seconds * 1000.0)

    hours = milliseconds // 3_600_000
    milliseconds -= hours * 3_600_000

    minutes = milliseconds // 60_000
    milliseconds -= minutes * 60_000

    seconds = milliseconds // 1_000
    milliseconds -= seconds * 1_000

    hours_marker = f"{hours:02d}:" if always_include_hours or hours > 0 else ""
    return f"{hours_marker}{minutes:02d}:{seconds:02d}{fractionalSeperator}{milliseconds:03d}"

# helper function to convert transcripts generated by whisper to .vtt file
def write_vtt(transcript: Iterator[dict], file: TextIO, maxLineWidth=None):
    print("WEBVTT\n", file=file)
    for segment in transcript:
        text = _processText(segment['text'], maxLineWidth).replace('-->', '->')

        print(
            f"{format_timestamp(segment['start'])} --> {format_timestamp(segment['end'])}\n"
            f"{text}\n",
            file=file,
            flush=True,
        )

# helper function to convert transcripts generated by whisper to .srt file
def write_srt(transcript: Iterator[dict], file: TextIO, maxLineWidth=None):
    """
    Write a transcript to a file in SRT format.
    Example usage:
        from pathlib import Path
        from whisper.utils import write_srt
        result = transcribe(model, audio_path, temperature=temperature, **args)
        # save SRT
        audio_basename = Path(audio_path).stem
        with open(Path(output_dir) / (audio_basename + ".srt"), "w", encoding="utf-8") as srt:
            write_srt(result["segments"], file=srt)
    """
    for i, segment in enumerate(transcript, start=1):
        text = _processText(segment['text'].strip(), maxLineWidth).replace('-->', '->')

        # write srt lines
        print(
            f"{i}\n"
            f"{format_timestamp(segment['start'], always_include_hours=True, fractionalSeperator=',')} --> "
            f"{format_timestamp(segment['end'], always_include_hours=True, fractionalSeperator=',')}\n"
            f"{text}\n",
            file=file,
            flush=True,
        )

def getSubs(segments: Iterator[dict], format: str, maxLineWidth: int=-1) -> str:
    segmentStream = StringIO()

    if format == 'vtt':
        write_vtt(segments, file=segmentStream, maxLineWidth=maxLineWidth)
    elif format == 'srt':
        write_srt(segments, file=segmentStream, maxLineWidth=maxLineWidth)
    else:
        raise Exception("Unknown format " + format)

    segmentStream.seek(0)
    return segmentStream.read()

#When only Video and Audio is available without Transcipt
def process_without_transcript(path_to_video_no_transcript):
    path_to_extracted_audio_file = os.path.join(path_to_video_no_transcript, 'audio.mp3')
    # extract mp3 audio file from mp4 video video file
    clip = VideoFileClip(path_to_video_no_transcript)
    clip.audio.write_audiofile(path_to_extracted_audio_file)
    model = whisper.load_model("small")
    options = dict(task="translate", best_of=1, language='en')
    results = model.transcribe(path_to_extracted_audio_file, **options)
    vtt = getSubs(results["segments"], "vtt")

    # path to save generated transcript of video1
    path_to_generated_trans = osp.join(path_to_video_no_transcript, 'generated_video1.vtt')
    # write transcription to file
    with open(path_to_generated_trans, 'w') as f:
        f.write(vtt)
