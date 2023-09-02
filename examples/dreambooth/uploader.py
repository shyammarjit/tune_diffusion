from googleapiclient.http import MediaFileUpload
from Google import Create_Service
import os, shutil
print("zipping")
output_dir = "/home/ai/.cache/huggingface/hub/models--stabilityai--stable-diffusion-xl-base-1.0"
archived = shutil.make_archive(os.path.basename(output_dir), 'zip', output_dir)
print("zip done")
print("uploading")
CLIENT_SECRET_FILE = 'Client_secret.json'
API_NAME ='drive'
API_VERSION = 'v3'
SCOPES = ['https://www.googleapis.com/auth/drive']

service = Create_Service(CLIENT_SECRET_FILE, API_NAME, API_VERSION, SCOPES)

# folder_id = '1IqHpR033dnKtgCTrtptj047-LNeKkmxd'
folder_id = "1Tt4dhZ8VYF1HMFsQm6ouefIvZrJ6Qk8d"
file_name = os.path.basename(archived)

file_metadata = {
    'name': file_name,
    'parents': [folder_id]
}

media = MediaFileUpload(archived, mimetype='application/octet-stream')

service.files().create(
    body = file_metadata,
    media_body = media,
    fields = 'id'
).execute()