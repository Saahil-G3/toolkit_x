host = "http://10.100.76.46:9020"
access_key_id = 'caib-pathadmin'
secret_key = 'DI2WC8g/ttSFf+02KN1ADtbeX0PX9fJaCzjJdsqN'

import boto3
from tqdm.auto import tqdm
from pathlib import Path
import load

def download_caib_slide(slide_path, bucket_name='caib-wsi', caib_keys_path=Path('caib-keys.pkl')):
    s3=get_s3_object()
    if slide_path.exists():
        print(f'Slide already downloaded.')
    else:
        
        if caib_keys_path.exists():
            keys=load.load_pickle(caib_keys_path)
        else:
            keys = get_keys_from_bucket(bucket_name)
            load.save_pickle(caib_keys_path, keys)
        
        print('Downloading Slide')
        for key in keys:
            if slide_path.stem in key:
                break
        download_file(bucket_name, key, slide_path)
    
def get_s3_object():   
    secure = True
    s3 = boto3.client('s3',aws_access_key_id=access_key_id, aws_secret_access_key=secret_key, use_ssl=secure, endpoint_url = host)
    return s3

def get_buckets(s3):
    
    response = s3.list_buckets()
    buckets = [bucket['Name'] for bucket in response['Buckets']]
    
    bucket_list = {}
    for index, bucket in enumerate(buckets):
        bucket_list.update({index:bucket})
    
    return(bucket_list)


def list_bucket_names():
    s3 = get_s3_object()
    buckets = get_buckets(s3)
    print("Available Buckets")
    for key, element in buckets.items():
        print(f'{key}:{element}')

def get_keys_from_bucket(bucket_name):
    
    s3 = get_s3_object()
    #buckets = get_buckets(s3)
    #print("Available Buckets")
    #for key, element in buckets.items():
    #    print(f'{key}:{element}')
    
    #bucket_index = input("Enter bucket index: ")
    
    #bucket_name = get_buckets(s3)[int(bucket_index)]
    
    all_object_keys = []
    paginator = s3.get_paginator('list_objects_v2')
    for page in paginator.paginate(Bucket=bucket_name):
        if 'Contents' in page:
            # Extract and append object keys to the list
            keys_in_page = [obj['Key'] for obj in page['Contents']]
            all_object_keys.extend(keys_in_page)
            
    print(f'Total objects: {len(all_object_keys)}')
    return(all_object_keys)

def download_file(bucket_name, object_key, local_file_path):
    
    s3 = get_s3_object()
    print('Starting Download')
    s3.download_file(bucket_name, object_key, local_file_path)
    print('Download Finished')

def find_key(query_string, keys):
    keys_found = []
    for key in keys:
        if query_string in key:
            #print(f'Key found: {key}')
            keys_found.append(key)

    if len(keys_found) >= 1:
        print(f"{len(keys_found)} Keys found")
        return(keys_found)
    else:
        print('No key(s) found')
        return []

def upload_file(bucket_name, upload_key, local_file_path):
    s3 = get_s3_object()
    print('Upload Started')
    s3.upload_file(local_file_path, bucket_name, upload_key)
    print('File Uploaded')
