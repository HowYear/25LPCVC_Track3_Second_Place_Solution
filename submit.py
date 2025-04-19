import torch
import qai_hub as hub
import requests
import numpy as np
from PIL import Image
import matplotlib 
import cv2 
import os 
from DepthAnythingV2.depth_anything_v2.dinov2_layers.replace_attention import replace_attention_with_v2
from DepthAnythingV2.depth_anything_v2.dpt import DepthAnythingV2

model_weight = './depth_anything_v2_metric_hypersim_vits.pth'

model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
}
width = 640 
height = 480 
encoder = 'vits' 
#########################################################################################################
# Step 0: 
torch_model = DepthAnythingV2(**model_configs[encoder])

torch_model.load_state_dict(torch.load(model_weight, map_location='cpu'), strict=False)
torch_model.eval()
torch_model = replace_attention_with_v2(torch_model)
#########################################################################################################
# Step 1: Trace model
input_shape = (1, 3, height, width)
example_input = torch.rand(input_shape)
traced_torch_model = torch.jit.trace(torch_model, example_input)
#########################################################################################################
# Step 2: Compile model
compile_job = hub.submit_compile_job(
    name=f'depth_anything_v2_{encoder}_350x350_flip',
    model=traced_torch_model,
    device=hub.Device("Snapdragon 8 Elite QRD"),
    input_specs=dict(image=input_shape),
    options="--target_runtime onnx",
)
# share the access permission of this compile job with our evaluation server 
compile_job.modify_sharing(add_emails=['lowpowervision@gmail.com'])

compiled_model = compile_job.get_target_model()
assert isinstance(compiled_model, hub.Model)
#########################################################################################################
# Step 3: Profile on cloud-hosted device
profile_job = hub.submit_profile_job(
    name=f'depth_anything_v2_{encoder}_350x350_flip',
    model=compiled_model,
    device=hub.Device("Snapdragon 8 Elite QRD"),
    # options="--onnx_options execution_mode=PARALLEL",
)

#########################################################################################################
# Step 4: Run inference on cloud-hosted device
sample_image_url = ("https://qaihub-public-assets.s3.us-west-2.amazonaws.com/apidoc/input_image1.jpg")
response = requests.get(sample_image_url, stream=True)
response.raw.decode_content = True
image = Image.open(response.raw).resize((width, height))
input_array = np.expand_dims(np.transpose(np.array(image, dtype=np.float32), (2, 0, 1)), axis=0).astype(np.float32) # (1, 3, 480, 640)

# Run inference using the on-device model on the input image
inference_job = hub.submit_inference_job(
    name=f'depth_anything_v2_{encoder}_350x350_flip',
    model=compiled_model,
    device=hub.Device("Snapdragon 8 Elite QRD"),
    inputs=dict(image=[input_array]),
)
on_device_output = inference_job.download_output_data()

### vis depth
cmap = matplotlib.colormaps.get_cmap('Spectral_r')
dd = on_device_output['output_0'][0][0]
depth = dd[0]
depth = (depth -np.min(depth))/(np.max(depth)-np.min(depth))
depth = depth * 255.0
depth = depth.astype(np.uint8)
cv2.imwrite(f'test_tea_{encoder}_gray.png', depth)
depth = (cmap(depth)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)
cv2.imwrite(f'test_tea_{encoder}.png', depth)
#########################################################################################################
