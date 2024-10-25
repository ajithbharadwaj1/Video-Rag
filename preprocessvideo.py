import os
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
from openai import AzureOpenAI

def isBase64(sb):
    try:
        if isinstance(sb, str):
                # If there's any unicode here, an exception will be thrown and the function will return false
                sb_bytes = bytes(sb, 'ascii')
        elif isinstance(sb, bytes):
                sb_bytes = sb
        else:
                raise ValueError("Argument must be string or bytes")
        return base64.b64encode(base64.b64decode(sb_bytes)) == sb_bytes
    except Exception:
            return False

class SeparatorStyle(Enum):
    """Different separator style."""
    SINGLE = auto()

@dataclasses.dataclass
class Conversation:
    """A class that keeps all conversation history"""
    system: str
    roles: List[str]
    messages: List[List[str]]
    map_roles: Dict[str, str]
    version: str = "Unknown"
    sep_style: SeparatorStyle = SeparatorStyle.SINGLE
    sep: str = "\n"    

    def _get_prompt_role(self, role):
        if self.map_roles is not None and role in self.map_roles.keys():
            return self.map_roles[role]
        else:
            return role
            
    def _build_content_for_first_message_in_conversation(self, first_message: List[str]):
        content = []
        if len(first_message) != 2:
            raise TypeError("First message in Conversation needs to include a prompt and a base64-enconded image!")
        
        prompt, b64_image = first_message[0], first_message[1]
        
        # handling prompt
        if prompt is None:
            raise TypeError("API does not support None prompt yet")
        content.append({
            "type": "text",
            "text": prompt
        })
        if b64_image is None:
            raise TypeError("API does not support text only conversation yet")
            
        # handling image
        if not isBase64(b64_image):
            raise TypeError("Image in Conversation's first message must be stored under base64 encoding!")
        
        content.append({
            "type": "image_url",
            "image_url": {
                "url": b64_image,
            }
        })
        return content

    def _build_content_for_follow_up_messages_in_conversation(self, follow_up_message: List[str]):

        if follow_up_message is not None and len(follow_up_message) > 1:
            raise TypeError("Follow-up message in Conversation must not include an image!")
        
        # handling text prompt
        if follow_up_message is None or follow_up_message[0] is None:
            raise TypeError("Follow-up message in Conversation must include exactly one text message")

        text = follow_up_message[0]
        return text
        
    def get_message(self):
        messages = self.messages
        api_messages = []
        for i, msg in enumerate(messages):
            role, message_content = msg
            if i == 0:                
                # get content for very first message in conversation
                content = self._build_content_for_first_message_in_conversation(message_content)
            else:
                # get content for follow-up message in conversation
                content = self._build_content_for_follow_up_messages_in_conversation(message_content)
                
            api_messages.append({
                "role": role,
                "content": content,
            })
        return api_messages

    # this method helps represent a multi-turn chat into as a single turn chat format
    def serialize_messages(self):
        messages = self.messages
        ret = ""
        if self.sep_style == SeparatorStyle.SINGLE:
            if self.system is not None and self.system != "":
                ret = self.system + self.sep
            for i, (role, message) in enumerate(messages):
                role = self._get_prompt_role(role)
                if message:
                    if isinstance(message, List):
                        # get prompt only
                        message = message[0]
                    if i == 0:
                        # do not include role at the beginning
                        ret += message
                    else:
                        ret += role + ": " + message
                    if i < len(messages) - 1:
                        # avoid including sep at the end of serialized message
                        ret += self.sep
                else:
                    ret += role + ":"
        else:
            raise ValueError(f"Invalid style: {self.sep_style}")

        return ret
    
    def append_message(self, role, message):
        if len(self.messages) == 0:
            # data verification for the very first message
            assert role == self.roles[0], f"the very first message in conversation must be from role {self.roles[0]}"
            assert len(message) == 2, f"the very first message in conversation must include both prompt and an image"
            prompt, image = message[0], message[1]
            assert prompt is not None, f"prompt must be not None"
            assert isBase64(image), f"image must be under base64 encoding"
        else:
            # data verification for follow-up message
            assert role in self.roles, f"the follow-up message must be from one of the roles {self.roles}"
            assert len(message) == 1, f"the follow-up message must consist of one text message only, no image"
            
        self.messages.append([role, message])

    def copy(self):
        return Conversation(
            system=self.system,
            roles=self.roles,
            messages=[[x,y] for x, y in self.messages],
            version=self.version,
            map_roles=self.map_roles,
        )

    def dict(self):
        return {
            "system": self.system,
            "roles": self.roles,
            "messages": [[x, y[0] if len(y) == 1 else y] for x, y in self.messages],
            "version": self.version,
        }

def download_video(video_url, path='/tmp/'):
    print(f'Getting video information for {video_url}')
    if not video_url.startswith('http'):
        return os.path.join(path, video_url)

    filepath = glob.glob(os.path.join(path, '*.mp4'))
    if len(filepath) > 0:
        return filepath[0]

    def progress_callback(stream: Stream, data_chunk: bytes, bytes_remaining: int) -> None:
        pbar.update(len(data_chunk))
    
    print("Downloading Youtube Video") 
    yt = YouTube(video_url, on_progress_callback=progress_callback)
    stream = yt.streams.filter(progressive=True, file_extension='mp4', res='720p').desc().first()
    if stream is None:
        stream = yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').desc().first()
    if not os.path.exists(path):
        os.makedirs(path)
    filepath = os.path.join(path, stream.default_filename)
    if not os.path.exists(filepath):   
        print('Downloading video from YouTube...')
        pbar = tqdm(desc='Downloading video from YouTube', total=stream.filesize, unit="bytes")
        stream.download(path)
        pbar.close()
    return filepath

def get_transcript_vtt(video_url, path='/tmp'):
    video_id = get_video_id_from_url(video_url)
    filepath = os.path.join(path,'captions.vtt')
    if os.path.exists(filepath):
        return filepath

    transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['en-GB', 'en'])
    formatter = WebVTTFormatter()
    webvtt_formatted = formatter.format_transcript(transcript)
    
    with open(filepath, 'w', encoding='utf-8') as webvtt_file:
        webvtt_file.write(webvtt_formatted)
    webvtt_file.close()

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

def encode_image(image_path_or_PIL_img):
    if isinstance(image_path_or_PIL_img, PIL.Image.Image):
        # this is a PIL image
        buffered = BytesIO()
        image_path_or_PIL_img.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')
    else:
        # this is a image_path
        with open(image_path_or_PIL_img, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
        
def lvlm_inference(prompt, image, max_tokens: int = 200, temperature: float = 0.95, top_p: float = 0.1):
    # prepare conversation
    conversation = prediction_guard_llava_conv.copy()
    conversation.append_message(conversation.roles[0], [prompt, image])
    return lvlm_inference_with_conversation(conversation, max_tokens=max_tokens, temperature=temperature, top_p=top_p)
    
prediction_guard_llava_conv = Conversation(
    system="",
    roles=("user", "assistant"),
    messages=[],
    version="Prediction Guard LLaVA enpoint Conversation v0",
    sep_style=SeparatorStyle.SINGLE,
    map_roles={
        "user": "USER", 
        "assistant": "ASSISTANT"
    }
)  

def lvlm_inference_with_conversation(conversation, max_tokens: int = 200, temperature: float = 0.95, top_p: float = 0.1):
    # get PredictionGuard client
    
    # get message from conversation
    messages = conversation.get_message()
    # call chat completion endpoint at Grediction Guard
    response = client.chat.completions.create(
        model="gai-sl-openai",
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p
        
    )
    return response['choices'][-1]['message']['content']
        
def lvlm_inference(prompt, image, max_tokens: int = 200, temperature: float = 0.95, top_p: float = 0.1):
    # prepare conversation
    conversation = prediction_guard_llava_conv.copy()
    conversation.append_message(conversation.roles[0], [prompt, image])
    return lvlm_inference_with_conversation(conversation, max_tokens=max_tokens, temperature=temperature, top_p=top_p)

def extract_and_save_frames_and_metadata_with_fps(
        path_to_video,  
        path_to_save_extracted_frames,
        path_to_save_metadatas,
        num_of_extracted_frames_per_second=1):
    
    # metadatas will store the metadata of all extracted frames
    metadatas = []

    # load video using cv2
    video = cv2.VideoCapture(path_to_video)
    
    # Get the frames per second
    fps = video.get(cv2.CAP_PROP_FPS)
    # Get hop = the number of frames pass before a frame is extracted
    hop = round(fps / num_of_extracted_frames_per_second) 
    curr_frame = 0
    idx = -1
    while(True):
        # iterate all frames
        ret, frame = video.read()
        if not ret: 
            break
        if curr_frame % hop == 0:
            idx = idx + 1
        
            # if the frame is extracted successfully, resize it
            image = maintain_aspect_ratio_resize(frame, height=350)
            # save frame as JPEG file
            img_fname = f'frame_{idx}.jpg'
            img_fpath = osp.join(
                            path_to_save_extracted_frames, 
                            img_fname
                        )
            cv2.imwrite(img_fpath, image)

            # generate caption using lvlm_inference
            lvlm_prompt = "Can you describe the image?"
            b64_image = encode_image(img_fpath)
            caption = lvlm_inference(lvlm_prompt, b64_image)
                
            # prepare the metadata
            metadata = {
                'extracted_frame_path': img_fpath,
                'transcript': caption,
                'video_segment_id': idx,
                'video_path': path_to_video,
            }
            metadatas.append(metadata)
        curr_frame += 1
        
    # save metadata of all extracted frames
    metadatas_path = osp.join(path_to_save_metadatas,'metadatas.json')
    with open(metadatas_path, 'w') as outfile:
        json.dump(metadatas, outfile)
    return metadatas
