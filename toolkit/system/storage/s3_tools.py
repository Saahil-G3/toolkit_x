import boto3
from pathlib import Path
from tqdm.auto import tqdm

from .data_io_tools import save_pickle

class S3:
    def __init__(
        self, endpoint_url, aws_access_key_id, aws_secret_access_key, use_ssl=True
    ):
        self._endpoint_url = endpoint_url
        self._aws_access_key_id = aws_access_key_id
        self._aws_secret_access_key = aws_secret_access_key
        self._use_ssl = use_ssl
        self._s3 = boto3.client(
            "s3",
            aws_access_key_id=self._aws_access_key_id,
            aws_secret_access_key=self._aws_secret_access_key,
            use_ssl=self._use_ssl,
            endpoint_url=self._endpoint_url,
        )
        self.bucket_list = self._set_bucket_list()
        self.bucket_keys = {}
        self.queried_keys = {}

    def get_keys_from_bucket(self, bucket_name, show_progress=True):
        self.bucket_keys.setdefault(bucket_name, [])
        paginator = self._s3.get_paginator("list_objects_v2")
        iterator = paginator.paginate(Bucket=bucket_name)

        iterator = tqdm(iterator, disable=not show_progress)

        for page in iterator:
            if "Contents" in page:
                self.bucket_keys[bucket_name].extend(
                    obj["Key"] for obj in page["Contents"]
                )

    def _set_bucket_list(self):
        response = self._s3.list_buckets()
        return {
            index: bucket["Name"] for index, bucket in enumerate(response["Buckets"])
        }

    def download_file(
        self, bucket_name: str, object_key: Path, folder: Path = Path(f"s3_files")
    ):
        object_key = Path(object_key)
        local_file_path = folder / object_key.name
        local_file_path.parent.mkdir(parents=True, exist_ok=True)
        self._s3.download_file(bucket_name, str(object_key), str(local_file_path))

    def upload_file(self, bucket_name:str, upload_key:str, local_file_path:str):
        self._s3.upload_file(local_file_path, bucket_name, upload_key)

    def find_key(self, query_string, bucket_name):
        if bucket_name not in self.bucket_keys.keys():
            s3.get_keys_from_bucket(bucket_name)
            
        keys_found = []
        for key in self.bucket_keys[bucket_name]:
            if query_string in key:
                keys_found.append(key)
    
        if len(keys_found) >= 1:
            self.queried_keys[query_string] = keys_found
        else:
            self.queried_keys[query_string] = "No keys found"