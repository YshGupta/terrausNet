from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
import io
from googleapiclient.http import MediaIoBaseDownload

import argparse
import os
import io
import base64
import numpy as np
import torch
from flask import Flask, render_template, request
from PIL import Image
from dataset_zenodo import Zenodo_dataset
from torchvision import transforms

from networks.vit_seg_modeling import VisionTransformer as ViT_seg
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
import torch.onnx

SCOPES = ['https://www.googleapis.com/auth/drive.readonly']

# Function to authenticate with Google Drive API
def authenticate_with_drive(client_id, client_secret):
    flow = InstalledAppFlow.from_client_secrets_file(
        'credentials.json', SCOPES)
    creds = flow.run_local_server(port=0)
    return creds

# Function to download a file from Google Drive
def download_file_from_drive(drive_service, file_id, file_name):
    request = drive_service.files().get_media(fileId=file_id)
    fh = io.FileIO(file_name, mode='wb')
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while not done:
        status, done = downloader.next_chunk()
        print(f'Download {int(status.progress() * 100)}%.')

client_id = '996393236200-nfnj0rdbajua8qfdpk5i2psm4el6us6u.apps.googleusercontent.com'
client_secret = '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'

# Authenticate with Google Drive API
credentials = authenticate_with_drive(client_id, client_secret)

# Build the Google Drive API service
drive_service = build('drive', 'v3', credentials=credentials)

# File ID of the file you want to download from Google Drive
file_id = '1mMh1GAiHl_fMYymxGTknWJ4qC_GPBmHX'

# Name to save the downloaded file locally
file_name = 'best_model_1.pth'

# Download the file from Google Drive
download_file_from_drive(drive_service, file_id, file_name)



parser = argparse.ArgumentParser()
parser.add_argument('--volume_path', type=str,
                    default='../data/Synapse/test_vol_h5', help='root dir for validation volume data')  # for acdc volume_path=root_dir
parser.add_argument('--dataset', type=str,
                    default='Synapse', help='experiment_name')
parser.add_argument('--num_classes', type=int,
                    default=2, help='output channel of network')
parser.add_argument('--list_dir', type=str,
                    default='./lists/lists_Synapse', help='list dir')

parser.add_argument('--max_iterations', type=int,default=20000, help='maximum epoch number to train')
parser.add_argument('--max_epochs', type=int, default=100, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=24,
                    help='batch_size per gpu')
parser.add_argument('--img_size', type=int, default=256, help='input patch size of network input')
parser.add_argument('--is_savenii', default=True, action="store_true", help='whether to save results during inference')

parser.add_argument('--n_skip', type=int, default=3, help='using number of skip-connect, default is num')
parser.add_argument('--vit_name', type=str, default='ViT-B_16', help='select one vit model')

parser.add_argument('--test_save_dir', type=str, default='predictions', help='saving prediction as nii!')
parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.01, help='segmentation network learning rate')
parser.add_argument('--seed', type=int, default=1234, help='random seed')
parser.add_argument('--vit_patches_size', type=int, default=16, help='vit_patches_size, default is 16')
args = parser.parse_args()


def inference(image, net):
    image = image.squeeze(0).cpu().detach().numpy()
    input = torch.from_numpy(image).unsqueeze(0).float()
    net.eval()
    with torch.no_grad():
        prediction, _, _, _, _ = net(input)
        prediction = prediction.squeeze()
        prediction = prediction.cpu().detach().numpy()
    pred = prediction.astype(np.float32)
    pred_binarized = np.zeros_like(pred)
    threshold = 0.5
    pred_binarized[pred > threshold] = 255
    output_image = Image.fromarray(pred_binarized.astype(np.uint8))
    output_image.save("ab.png", "PNG")
    return output_image


dataset_config = {
    'Synapse': {
        'Dataset': Zenodo_dataset, #Synapse_dataset,
        'volume_path': '../data/Synapse/test_vol_h5',
        'list_dir': './lists/lists_Synapse',
        'num_classes': 1,
        'z_spacing': 1,
    },
}
dataset_name = args.dataset
args.num_classes = dataset_config[dataset_name]['num_classes']
args.volume_path = dataset_config[dataset_name]['volume_path']
args.Dataset = dataset_config[dataset_name]['Dataset']
args.list_dir = dataset_config[dataset_name]['list_dir']
args.z_spacing = dataset_config[dataset_name]['z_spacing']
args.is_pretrain = True

# name the same snapshot defined in train script!
args.exp = 'TU_' + dataset_name + str(args.img_size)
snapshot_path = ""

config_vit = CONFIGS_ViT_seg['R50-ViT-B_16']
config_vit.n_classes = args.num_classes
config_vit.n_skip = args.n_skip
config_vit.patches.size = (args.vit_patches_size, args.vit_patches_size)
if args.vit_name.find('R50') !=-1:
    config_vit.patches.grid = (int(args.img_size/args.vit_patches_size), int(args.img_size/args.vit_patches_size))
net = ViT_seg(config_vit, img_size=args.img_size, num_classes=config_vit.n_classes)

snapshot = os.path.join('best_model_1.pth')
net.load_state_dict(torch.load(snapshot, map_location=torch.device('cpu')))

transform = transforms.ToTensor()

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'image' not in request.files:
        return 'No file part'

    file = request.files['image']

    if file.filename == '':
        return 'No selected file'

    img = Image.open(file)
    # print(img)
    transform = transforms.ToTensor()
    output_image = inference(transform(img), net)
    output_image_base64 = image_to_base64(output_image)

    # Display the resultant image
    return f'''
        <h2>Uploaded Image:</h2>
        <img src="data:image/jpeg;base64,{output_image_base64}" alt="Uploaded Image">
    '''

def image_to_base64(image):
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str

if __name__ == '__main__':
    app.run(debug=True,port= 5000)
