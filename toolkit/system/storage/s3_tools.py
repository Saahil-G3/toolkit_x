import boto3
from pathlib import Path
from tqdm.auto import tqdm

from toolkit.system.logging_tools import Logger
logger = Logger(name="s3_tools").get_logger()

from .data_io_tools import save_pickle


def get_s3_object(endpoint_url, aws_access_key_id, aws_secret_access_key, use_ssl=True):
    s3 = boto3.client(
        "s3",
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        use_ssl=use_ssl,
        endpoint_url=endpoint_url,
    )
    return s3

from toolkit.system.storage.data_io_tools import save_pickle, load_pickle

class S3:
    def __init__(self, s3_object):
        self._s3 = s3_object
        self.bucket_list = self._set_bucket_list()
        self.bucket_keys = {}
        self.queried_keys = {}

    def get_keys_from_bucket(self, bucket_name, show_progress=True, replace_keys=False):
        keys_path = Path(f"s3/bucket_keys/{bucket_name}.pkl")
        keys_path.parent.mkdir(exist_ok=True, parents=True)
        if keys_path.exists() and not replace_keys:
            self.bucket_keys[bucket_name] = load_pickle(keys_path)
        else:
            self.bucket_keys.setdefault(bucket_name, [])
            paginator = self._s3.get_paginator("list_objects_v2")
            iterator = paginator.paginate(Bucket=bucket_name)
    
            iterator = tqdm(iterator, disable=not show_progress)
    
            for page in iterator:
                if "Contents" in page:
                    self.bucket_keys[bucket_name].extend(
                        obj["Key"] for obj in page["Contents"]
                    )
            
            save_pickle(data=self.bucket_keys[bucket_name], path=keys_path, replace=replace_keys)

    def _set_bucket_list(self):
        response = self._s3.list_buckets()
        return {
            index: bucket["Name"] for index, bucket in enumerate(response["Buckets"])
        }

    def download_file(
        self,
        bucket_name: str,
        object_key: str,
        local_file_path: str,
    ):

        if Path(local_file_path).exists():
            message = f"File already exists at {local_file_path}"
            logger.info(message)
            return

        Path(local_file_path).parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"Started downloading {Path(object_key).name}.")
        self._s3.download_file(Bucket=bucket_name, Key=object_key, Filename=local_file_path)
        logger.info(f"File downloaded at {local_file_path}.")

    def upload_file(self, bucket_name: str, upload_key: str, local_file_path: str):
        self._s3.upload_file(local_file_path, bucket_name, upload_key)

    def find_key(self, query_string, bucket_name):
        if bucket_name not in self.bucket_keys.keys():
            self.get_keys_from_bucket(bucket_name)

        keys_found = []
        for key in self.bucket_keys[bucket_name]:
            if query_string in key:
                keys_found.append(key)

        if len(keys_found) >= 1:
            self.queried_keys[query_string] = keys_found
            logger.info(f"Found keys for {query_string}.")
        else:
            logger.info(f"No keys found for {query_string}.")
