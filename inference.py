import argparse
import math
import os
import cv2
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import openvino as ov
import audio
from models import Wav2Lip
from batch_face import RetinaFace
# from retinaface import RetinaFace as retina_face
from time import time, sleep
import pyaudio
from PIL import Image, ImageTk

parser = argparse.ArgumentParser(description='Inference code to lip-sync videos in the wild using Wav2Lip models')

parser.add_argument('--checkpoint_path', type=str, default="./Wav2Lip/checkpoints/wav2lip_gan.pth",
                    help='Name of saved checkpoint to load weights from', required=False)
parser.add_argument('--face', type=str, default="../Wav2Lip_stream/ip/test.jpg",
                    help='Filepath of video/image that contains faces to use', required=False)
parser.add_argument('--audio', type=str, 
                    help='Filepath of video/audio file to use as raw audio source', required=False)
parser.add_argument('--outfile', type=str, help='Video path to save result. See default for an e.g.', 
                    default='results/result_voice.mp4')
parser.add_argument('--static', type=bool, 
                    help='If True, then use only first video frame for inference', default=False)
parser.add_argument('--fps', type=float, help='Can be specified only if input is a static image (default: 25)', 
                    default=15., required=False)
parser.add_argument('--pads', nargs='+', type=int, default=[0, 10, 0, 0], 
                    help='Padding (top, bottom, left, right). Please adjust to include chin at least')
parser.add_argument('--wav2lip_batch_size', type=int, help='Batch size for Wav2Lip model(s)', default=8)
parser.add_argument('--resize_factor', default=1, type=int,
                    help='Reduce the resolution by this factor. Sometimes, best results are obtained at 480p or 720p')
parser.add_argument('--out_height', default=480, type=int,
                    help='Output video height. Best results are obtained at 480 or 720')
parser.add_argument('--crop', nargs='+', type=int, default=[0, -1, 0, -1],
                    help='Crop video to a smaller region (top, bottom, left, right). Applied after resize_factor and rotate arg. ' 
                    'Useful if multiple face present. -1 implies the value will be auto-inferred based on height, width')
parser.add_argument('--box', nargs='+', type=int, default=[-1, -1, -1, -1], 
                    help='Specify a constant bounding box for the face. Use only as a last resort if the face is not detected.'
                    'Also, might work only if the face is not moving around much. Syntax: (top, bottom, left, right).')
parser.add_argument('--rotate', default=False, action='store_true',
                    help='Sometimes videos taken from a phone can be flipped 90deg. If true, will flip video right by 90deg.'
                    'Use if you get a flipped result, despite feeding a normal looking video')
parser.add_argument('--nosmooth', default=False, action='store_true', help='Prevent smoothing face detections over a short temporal window')

class Wav2LipInference:
    def __init__(self, args) -> None:
        self.args = args
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu' # Device selection: Uses GPU ('cuda') if available, otherwise defaults to CPU.
        self.CHUNK = 1024  # Number of audio frames per buffer during audio capture.
        self.FORMAT = pyaudio.paInt16 # Audio format: 16-bit PCM
        self.CHANNELS = 1  # Number of audio channels
        self.RATE = 16000  # sample rate (samples per second)
        self.RECORD_SECONDS = 0.5  # Duration of audio recording per capture
        self.mel_step_size = 16 # Step size for mel spectrogram processing
        self.model = self.load_model()
        self.detector = self.load_batch_face_model()
        self.face_detect_cache_result = None


    def get_smoothened_boxes(self, boxes, T):
        for i in range(len(boxes)):
            if i + T > len(boxes):
                window = boxes[len(boxes) - T:]
            else:
                window = boxes[i : i + T]
            boxes[i] = np.mean(window, axis=0)
        return boxes

    def face_detect(self, images):
        results = []
        pady1, pady2, padx1, padx2 = self.args.pads
        s = time()
        for image, rect in zip(images, self.face_rect(images)):
            if rect is None:
                print("Face was not detected...")
                continue
                cv2.imwrite('temp/faulty_frame.jpg', image) # check this frame where the face was not detected.
                raise ValueError('Face not detected! Ensure the video contains a face in all the frames.')
            y1 = max(0, rect[1] - pady1)
            y2 = min(image.shape[0], rect[3] + pady2)
            x1 = max(0, rect[0] - padx1)
            x2 = min(image.shape[1], rect[2] + padx2)
            results.append([x1, y1, x2, y2])

        print('face detect time:', time() - s)
        boxes = np.array(results)
        if not self.args.nosmooth: boxes = self.get_smoothened_boxes(boxes, T=5)
        results = [[image[y1: y2, x1:x2], (y1, y2, x1, x2)] for image, (x1, y1, x2, y2) in zip(images, boxes)]
        return results

    def datagen(self, frames, mels):

        img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []
        if self.args.box[0] == -1:
            if not self.args.static:
                face_det_results = self.face_detect(frames) # BGR2RGB for CNN face detection
            else:
                face_det_results = self.face_detect_cache_result # use cached result face_detect([frames[0]])
        else:
            print('Using the specified bounding box instead of face detection...')
            y1, y2, x1, x2 = self.args.box
            face_det_results = [[f[y1: y2, x1:x2], (y1, y2, x1, x2)] for f in frames]
        for i, m in enumerate(mels):        
            idx = 0 if self.args.static else i%len(frames)
            frame_to_save = frames[idx].copy()
            face, coords = face_det_results[idx].copy()
            face = cv2.resize(face, (self.args.img_size, self.args.img_size))
            img_batch.append(face)
            mel_batch.append(m)
            frame_batch.append(frame_to_save)
            coords_batch.append(coords)

            if len(img_batch) >= self.args.wav2lip_batch_size:
                img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)
                img_masked = img_batch.copy()
                img_masked[:, self.args.img_size//2:] = 0
                img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
                mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])
                yield img_batch, mel_batch, frame_batch, coords_batch
                img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

        # if there are any other batches
        if len(img_batch) > 0:
            
            img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)
            img_masked = img_batch.copy()
            img_masked[:, self.args.img_size//2:] = 0
            img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
            mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

            yield img_batch, mel_batch, frame_batch, coords_batch

    def load_model(self):
        if self.device == 'cpu':
            return self.load_openvino_model()
        else:
            return self.load_wav2lip_model(self.args.checkpoint_path)
        
    def load_openvino_model(self):
        #Load OpenVINO model
        print("Calling OpenVINO model for wav2lip inference")
        core = ov.Core()
        avail_devices = core.available_devices
        print(avail_devices[0])
        model = core.read_model(model="./wav2lip_openvino_model.xml")
        compiled_model = core.compile_model(model=model, device_name=avail_devices[0])
        return compiled_model

    def load_wav2lip_model(self, checkpoint_path):
        model = Wav2Lip()
        print("Load checkpoint from: {}".format(checkpoint_path))
        checkpoint = self.load_model_weights(checkpoint_path)
        s = checkpoint["state_dict"]
        new_s = {k.replace('module.', ''): v for k, v in s.items()}
        model.load_state_dict(new_s)
        model = model.to(self.device)
        return model.eval()
    
    def load_model_weights(self, checkpoint_path):
        if self.device == 'cuda':
            checkpoint = torch.load(checkpoint_path)
        else:
            checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
        return checkpoint

    def load_batch_face_model(self):
        if self.device == 'cpu':
            return RetinaFace(gpu_id=-1, model_path="checkpoints/mobilenet.pth", network="mobilenet")
        else:
            return RetinaFace(gpu_id=0, model_path="checkpoints/mobilenet.pth", network="mobilenet")

    def face_rect(self, images):
        # Define the batch size for processing faces in chunks.
        face_batch_size = 64 * 8
        num_batches = math.ceil(len(images) / face_batch_size)
        prev_rect = None
        for i in range(num_batches):
            # Get the current batch of images.
            batch = images[i * face_batch_size: (i + 1) * face_batch_size]
            # Detect faces in the current batch.
            faces = self.detector(batch)
            for face in faces:
                if face:
                    # Get the bounding box coordinates of the first detected face.
                    box, _, _ = face[0]
                    prev_rect = tuple(map(int, box))
                yield prev_rect

    def record_audio(self, audio_stream):
        stime = time()
        print("Recording audio")
        frames = []
        for i in range(0, int(self.RATE / self.CHUNK * self.RECORD_SECONDS)):
            frames.append(audio_stream.read(self.CHUNK, exception_on_overflow=False))
        print("Recording time: ", time() - stime)
        audio_data = np.frombuffer(b''.join(frames), dtype=np.int16)
        # Check if audio data has sufficient amplitude
        if np.max(np.abs(audio_data)) < 500:
            print("Warning: Recorded audio is too quiet.")
        print(f"Recorded audio length: {len(audio_data)}")
        return audio_data

    def get_mel_chunks(self, audio_data):
        stime = time()
        mel = audio.melspectrogram(audio_data)
        print(mel.shape)
        if np.isnan(mel.reshape(-1)).sum() > 0:
            raise ValueError('Mel contains nan! Using a TTS voice? Add a small epsilon noise to the wav file and try again')
        mel_chunks = []
        mel_idx_multiplier = 80./self.args.fps
        i = 0
        while 1:
            start_idx = int(i * mel_idx_multiplier)
            if start_idx + self.mel_step_size > len(mel[0]):
                mel_chunks.append(mel[:, len(mel[0]) - self.mel_step_size:])
                break
            mel_chunks.append(mel[:, start_idx : start_idx + self.mel_step_size])
            i += 1
        print("Length of mel chunks: {}".format(len(mel_chunks)))
        return mel_chunks

def update_frames(full_frames, audio_stream, inference_obj):
    stime = time()
    audio_data = inference_obj.record_audio(audio_stream)
    mel_chunks = inference_obj.get_mel_chunks(audio_data)
    print(f"Time to process audio input {time()-stime}")

    full_frames = full_frames[:len(mel_chunks)]
    wav2lip_batch_size = inference_obj.args.wav2lip_batch_size
    gen = inference_obj.datagen(full_frames.copy(), mel_chunks.copy())

    s = time()
    for i, (img_batch, mel_batch, frames, coords) in enumerate(tqdm(gen, total=int(np.ceil(float(len(mel_chunks)) / wav2lip_batch_size)))):
        if inference_obj.device == 'cpu':
            img_batch = np.transpose(img_batch, (0, 3, 1, 2))
            mel_batch = np.transpose(mel_batch, (0, 3, 1, 2))
            pred = inference_obj.model([mel_batch, img_batch])['output']
        else:
            img_batch = torch.FloatTensor(np.transpose(img_batch, (0, 3, 1, 2))).to(inference_obj.device)
            mel_batch = torch.FloatTensor(np.transpose(mel_batch, (0, 3, 1, 2))).to(inference_obj.device)
            with torch.no_grad():
                pred = inference_obj.model(mel_batch, img_batch)

        pred = pred.transpose(0, 2, 3, 1) * 255.
        for p, f, c in zip(pred, frames, coords):
            y1, y2, x1, x2 = c
            p = cv2.resize(p.astype(np.uint8), (x2 - x1, y2 - y1))
            f[y1:y2, x1:x2] = p
            # Convert frame to RGB before yielding
            rgb_frame = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
            # Return the RGB frame for display in Streamlit
            yield rgb_frame

def main(image_path, flag):

    args = parser.parse_args()
    args.img_size = 96
    args.face = image_path
    inference_obj = Wav2LipInference(args)

    if os.path.isfile(args.face) and args.face.split('.')[-1] in ['jpg', 'png', 'jpeg']:
        args.static = True
    if not os.path.isfile(args.face):
        raise ValueError('--face argument must be a valid path to video/image file')
    elif args.face.split('.')[-1] in ['jpg', 'png', 'jpeg']:
        full_frames = [cv2.imread(args.face)]
        fps = args.fps
    else:
        video_stream = cv2.VideoCapture(args.face)
        fps = video_stream.get(cv2.CAP_PROP_FPS)
        print('Reading video frames')
        full_frames = []
        while 1:
            still_reading, frame = video_stream.read()
            if not still_reading:
                video_stream.release()
                break
            if args.resize_factor > 1:
                frame = cv2.resize(frame, (frame.shape[1]//args.resize_factor, frame.shape[0]//args.resize_factor))
            aspect_ratio = frame.shape[1] / frame.shape[0]
            frame = cv2.resize(frame, (int(args.out_height * aspect_ratio), args.out_height))

            if args.rotate:
                frame = cv2.rotate(frame, cv2.cv2.ROTATE_90_CLOCKWISE)
            y1, y2, x1, x2 = args.crop
            if x2 == -1: x2 = frame.shape[1]
            if y2 == -1: y2 = frame.shape[0]
            frame = frame[y1:y2, x1:x2]
            full_frames.append(frame)

    print ("Number of frames available for inference: "+str(len(full_frames)))

    p = pyaudio.PyAudio()
    stream = p.open(format=inference_obj.FORMAT,
                    channels=inference_obj.CHANNELS,
                    rate=inference_obj.RATE,
                    input=True,
                    frames_per_buffer=inference_obj.CHUNK)
    
    inference_obj.face_detect_cache_result = inference_obj.face_detect([full_frames[0]])
    while True:
        if not flag:
            stream.stop_stream()
            stream.close()
            p.terminate()
            return 
        yield from update_frames(full_frames, stream, inference_obj)