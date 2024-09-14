import os
import sys
from tempfile import NamedTemporaryFile
from urllib.request import urlopen
from urllib.parse import unquote, urlparse
from urllib.error import HTTPError
from zipfile import ZipFile
import tarfile
import shutil
import subprocess


CHUNK_SIZE = 40960
DATA_SOURCE_MAPPING = 'isic2018-challenge-task1-data-segmentation:https%3A%2F%2Fstorage.googleapis.com%2Fkaggle-data-sets%2F775382%2F1335230%2Fbundle%2Farchive.zip%3FX-Goog-Algorithm%3DGOOG4-RSA-SHA256%26X-Goog-Credential%3Dgcp-kaggle-com%2540kaggle-161607.iam.gserviceaccount.com%252F20240914%252Fauto%252Fstorage%252Fgoog4_request%26X-Goog-Date%3D20240914T083439Z%26X-Goog-Expires%3D259200%26X-Goog-SignedHeaders%3Dhost%26X-Goog-Signature%3D213d35cb98d1a9012ebc95c069c7d7d98fcb1b9c0e38b8ecdc8b4a5593a1adfa38b43d0f16ed9f9588812963310c3e3bf6e0d1e1668d5f69034006d0d054baa7def233ef669420696b81a9058b2ef5c6b20344c6d71fd78367a9bcd6987cefab76fd6346dcae6a9e4c29ab4d38c4e1d62bb1268d447a29ca5d4c57ca2673691dfe260c5445cb1a95096b531d71b08c8f488b3cf7e168086391f96129a8f30b87032436f487843f0f73d67edfb26d36ba572edca4535fc7a57cff6a6603df4bfe14187b86ae47ec86f5816e158be2408ecae7017c59d35c1dcb688c033c818d5d7cb498bc773f8a0ad7d07c7e56ad813c87c8ae8a8978c004abba7f6d51b772ad'

KAGGLE_INPUT_PATH='/kaggle/input'
KAGGLE_WORKING_PATH='/kaggle/working'
KAGGLE_SYMLINK='kaggle'

# !umount /kaggle/input/ 2> /dev/null
if os.name != 'nt':
    subprocess.run(['umount', '/kaggle/input/'], stderr=subprocess.DEVNULL)

shutil.rmtree('/kaggle/input', ignore_errors=True)
os.makedirs(KAGGLE_INPUT_PATH, 0o777, exist_ok=True)
os.makedirs(KAGGLE_WORKING_PATH, 0o777, exist_ok=True)

try:
    os.symlink(KAGGLE_INPUT_PATH, os.path.join("..", 'input'), target_is_directory=True)
except FileExistsError:
    pass
try:
    os.symlink(KAGGLE_WORKING_PATH, os.path.join("..", 'working'), target_is_directory=True)
except FileExistsError:
    pass

for data_source_mapping in DATA_SOURCE_MAPPING.split(','):
    directory, download_url_encoded = data_source_mapping.split(':')
    download_url = unquote(download_url_encoded)
    filename = urlparse(download_url).path
    destination_path = os.path.join(KAGGLE_INPUT_PATH, directory)
    try:
        with urlopen(download_url) as fileres, NamedTemporaryFile() as tfile:
            total_length = fileres.headers['content-length']
            print(f'Downloading {directory}, {total_length} bytes compressed')
            dl = 0
            data = fileres.read(CHUNK_SIZE)
            while len(data) > 0:
                dl += len(data)
                tfile.write(data)
                done = int(50 * dl / int(total_length))
                sys.stdout.write(f"\r[{'=' * done}{' ' * (50-done)}] {dl} bytes downloaded")
                sys.stdout.flush()
                data = fileres.read(CHUNK_SIZE)
            if filename.endswith('.zip'):
                with ZipFile(tfile) as zfile:
                    zfile.extractall(destination_path)
                    print(f'\nDownloaded and uncompressed: {directory}')
    except HTTPError as e:
        print(f'Failed to load (likely expired) {download_url} to path {destination_path}')
    except OSError as e:
        print(f'Failed to load {download_url} to path {destination_path}')

print('Data source import complete.')


i = 0
for dirname, _, filenames in os.walk(KAGGLE_INPUT_PATH):
    for filename in filenames:
        i+=1
        print(os.path.join(dirname, filename))

print(f'{i} files found in total.')