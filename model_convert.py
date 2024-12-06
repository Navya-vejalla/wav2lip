import torch
import os
from openvino.runtime import Core, save_model
from openvino import convert_model

from models import Wav2Lip
import numpy as np

device = "cpu"

def convert_pytorch_to_openvino(checkpoint_path, onnx_path):
    # Load your PyTorch model
    model = Wav2Lip()  # Ensure to initialize your model correctly
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()  # Set to evaluation mode

    # Create dummy input for exporting
    batch_size = 128
    img_batch = np.random.rand(batch_size, 6, 96, 96)  # Adjust as needed
    mel_batch = np.random.rand(batch_size, 1, 80, 16)  # Adjust as needed
    
    img_batch_tensor = torch.FloatTensor(img_batch).to(device)
    mel_batch_tensor = torch.FloatTensor(mel_batch).to(device)

    # Export to ONNX format
    print("Exporting to ONNX format...")
    torch.onnx.export(model, 
                      (mel_batch_tensor, img_batch_tensor),  # Using mel_batch and img_batch
                      onnx_path, 
                      export_params=True, 
                      opset_version=11, 
                      do_constant_folding=True, 
                      input_names=['audio_sequences', 'face_sequences'],  # New input names
                      output_names=['output'],  # Specify output names
                      dynamic_axes={'audio_sequences': {0: 'batch_size', 1: 'time_size'},  # Variable axes for mel input
                                    'face_sequences': {0: 'batch_size', 1: 'channel'},  # Variable axes for img input
                                    'output': {0: 'batch_size'}})  # Variable batch size for output

    print(f"ONNX model exported to {onnx_path}")

    #onnx model to openvino model
    core = Core()
    devices = core.available_devices
    print(devices[0])

    model_onnx = core.read_model(model=onnx_path)
    #compiled_model_onnx = core.compile_model(model=model_onnx, device_name=devices[0])

    save_model(model_onnx, output_model="wav2lip_openvino_model.xml")
    # model = convert_model(onnx_path)
    # save_model(model, 'openvino_model/wav2lip_model.xml')
    print("OpenVINO model saved")

if __name__ == '__main__':
    checkpoint_path = 'checkpoints/wav2lip_gan.pth'  # Path to your PyTorch checkpoint
    onnx_path = 'wav2lip_model.onnx'  # Path to save the ONNX model
    convert_pytorch_to_openvino(checkpoint_path, onnx_path)
