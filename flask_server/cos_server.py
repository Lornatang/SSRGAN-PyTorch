# Copyright 2020 Dakewe Biotech Corporation. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
from qcloud_cos import CosConfig
from qcloud_cos import CosS3Client
from ssrgan.dataset import check_image_file


class COS(object):
    def __init__(self):
        self.Region = "Your Region"
        self.SecretId = "Your SecretId"
        self.SecretKey = "Your SecretKey"
        self.Scheme = "https"
        self.Bucket = "Your Bucket"

        self.config = CosConfig(Region=self.Region,
                                SecretId=self.SecretId,
                                SecretKey=self.SecretKey,
                                Scheme=self.Scheme)
        self.client = CosS3Client(self.config)

    def get_all_urls(self):
        client = self.client  # init client API.
        bucket = self.Bucket
        url_lists = []  # Save all url address.
        marker = ""  # start index from 0.

        while True:
            response = client.list_objects(Bucket=bucket, Marker=marker)
            # Get corresponding image files.
            for index in range(len(response["Contents"])):
                Key = response["Contents"][index]["Key"]
                # Filtering long url and non image url.
                if len(Key) <= 50 and check_image_file(Key):
                    url_lists.append(Key)

            if response["IsTruncated"] == "false":
                break
            marker = response["NextMarker"]

        # Sort by the latest time.
        url_lists.sort(reverse=True)

        return url_lists

    def download_file(self, file_path, cos_path):
        """Download file from cos url address.
        Files less than or equal to 20MB are simply downloaded, and files larger than 20MB are downloaded continuously.

        Args:
            file_path (string): Download the file to save the address.
            cos_path (string): COS url address.
        """
        client = self.client  # init client API.
        response = client.get_object(
            Bucket=self.Bucket,
            Key=cos_path,
        )
        response["Body"].get_stream_to_file(file_path)

    def upload_file(self, stream, cos_path):
        """Upload file to COS URL.
        Files less than or equal to 20MB are uploaded simply, and files larger than 20MB are uploaded in blocks.

        Args:
            stream (bytes): The uploaded file content is of file stream or byte stream type.
            cos_path (string): COS url address.
        """
        client = self.client  # init client API.
        client.put_object(
            Bucket=self.Bucket,  # Bucket storage name.
            Body=stream,
            Key=cos_path  # Block upload path name.
        )
