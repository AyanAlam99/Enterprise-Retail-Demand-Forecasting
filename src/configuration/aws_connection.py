import os 
import boto3
from src.constants import AWS_SECRET_ACCESS_KEY_ENV_KEY , AWS_ACCESS_KEY_ID_ENV_KEY , REGION_NAME

class S3Client : 
    s3_client = None 
    s3_resources = None 

    def __init__(self, region_name = REGION_NAME ) : 
        '''
        This class gets aws credentials and creates a connection with s3 bucket 
        '''

        if S3Client.s3_resources ==None  or S3Client.s3_client == None : 
            __access_key_id = os.getenv(AWS_ACCESS_KEY_ID_ENV_KEY)
            __secret_access_key = os.getenv(AWS_SECRET_ACCESS_KEY_ENV_KEY)
            if __access_key_id is None : 
                raise Exception(f"{AWS_ACCESS_KEY_ID_ENV_KEY} is not set")
            if __secret_access_key is None : 
                raise Exception(f"{AWS_SECRET_ACCESS_KEY_ENV_KEY} is not set ")

            S3Client.s3_resources= boto3.resource(
                's3',
                aws_access_key_id = __access_key_id,
                aws_secret_key_id = __secret_access_key,
                region_name = region_name 
            )
            S3Client.s3_client= boto3.client(
                's3',
                aws_access_key_id = __access_key_id,
                aws_secret_key_id = __secret_access_key,
                region_name = region_name 
            )

        self.s3_resources = S3Client.s3_resources
        self.s3_client = S3Client.s3_client

