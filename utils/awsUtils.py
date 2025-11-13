import boto3
from botocore.exceptions import ClientError
from copy import deepcopy
import ast
import os

from utils.alertSys import CentralizedSlackAlert
from utils.utils import json_parser, json_exporter, yaml_parser


class AWS:
    def __init__(self, env: str, configFilePath: str, secret_name: str):

        secretDict = {
            "VM": {
                "slackChannelUrls": "DEVELOPMENT_CENTRALISED_ALERT_SYSTEM_SLACK_CHANNEL_URLS",
                "slackUserIds": "DEVELOPMENT_CENTRALISED_ALERT_SYSTEM_SLACK_USER_IDS",
            },
            "development": {
                "slackChannelUrls": "DEVELOPMENT_CENTRALISED_ALERT_SYSTEM_SLACK_CHANNEL_URLS",
                "slackUserIds": "DEVELOPMENT_CENTRALISED_ALERT_SYSTEM_SLACK_USER_IDS",
            },
            "staging": {
                "slackChannelUrls": "STAGING_CENTRALISED_ALERT_SYSTEM_SLACK_CHANNEL_URLS",
                "slackUserIds": "STAGING_CENTRALISED_ALERT_SYSTEM_SLACK_USER_IDS",
            },
            "production": {
                "slackChannelUrls": "PLUANG_CENTRALISED_ALERT_SYSTEM_SLACK_CHANNEL_URLS_PRODUCTION",
                "slackUserIds": "PLUANG_CENTRALISED_ALERT_SYSTEM_SLACK_USER_IDS_PRODUCTION",
            },
        }

        self.env = env
        self.CONFIG = yaml_parser(configFilePath)
        self.secretMap = json_parser(f"./config/secret-map/{env}.json")
        self.cloud = "AWS"
        self.credentialViaAwsKey = env in ["VM", "development"]

        self.aws_client_init()
        if not self.credentialViaAwsKey:
            self.load_env_as_dict()
        
        AWS_SECRET_CONFIG = {}
        if secret_name:
            AWS_SECRET_CONFIG = self.get_aws_secret_manager_value(key=secret_name)
        AWS_ACCESS_KEY = AWS_SECRET_CONFIG.get("AWS_ACCESS_KEY", "")
        AWS_SECRET_KEY = AWS_SECRET_CONFIG.get("AWS_SECRET_KEY", "")
        
        if AWS_ACCESS_KEY and AWS_SECRET_KEY:
            self.aws_client_init(AWS_ACCESS_KEY, AWS_SECRET_KEY)
        
        self.slackChannelNameUrlDict = self.get_aws_secret_manager_value(
            secretDict[env]["slackChannelUrls"]
        )
        self.slackIdDict = self.get_aws_secret_manager_value(
            secretDict[env]["slackUserIds"]
        )

    def load_env_as_dict(self):
        """
        Load environment variables into a dictionary, attempting to parse string values as dictionaries.
        This function is useful for converting environment variables that are stored as strings.

        Side Effects:
            Sets the `self.envVariables` attribute to a dictionary containing the parsed environment variables.

        Returns:
            None
        """

        # Function to try to evaluate a string as a dictionary
        # If the string is not a valid dictionary, it will return the original string
        # This is useful for converting environment variables that are stored as strings
        def try_eval_dict(value):
            try:
                parsed = ast.literal_eval(value)
                if isinstance(parsed, dict):
                    return {
                        k: try_eval_dict(v) if isinstance(v, str) else v
                        for k, v in parsed.items()
                    }
            except Exception:
                pass
            return value

        result = {}
        for key, value in os.environ.items():
            result[key] = try_eval_dict(value)
        self.envVariables = result

    def aws_client_init(self, aws_access_key: str = "", aws_secret_key: str = ""):
        """
        Initializes the AWS S3 client using credentials from a specified JSON file. This function
        extracts the necessary access key and secret key from the file, and sets up the S3 client
        using the `boto3` library. The JSON file path is obtained from `CONFIG["filePathAwsKey"]`.

        Raises:
            KeyError: If required credentials are not found in the JSON file.
        """
        # self.awsKey = json_parser(self.CONFIG["filePathAwsKey"])
        self.awsKey = {}
        if aws_access_key:
            self.awsKey["aws_access_key_id"] = aws_access_key
        if aws_secret_key:
            self.awsKey["aws_secret_access_key"] = aws_secret_key

        # Build client kwargs - only add credentials if they exist
        s3_client_kwargs = {}
        secrets_client_kwargs = {
            "service_name": "secretsmanager",
            "region_name": self.CONFIG["awsRegionName"]
        }
        
        # Only add credentials if both key and secret are provided
        if aws_access_key and aws_secret_key:
            s3_client_kwargs["aws_access_key_id"] = aws_access_key
            s3_client_kwargs["aws_secret_access_key"] = aws_secret_key
            secrets_client_kwargs["aws_access_key_id"] = aws_access_key
            secrets_client_kwargs["aws_secret_access_key"] = aws_secret_key

        self.awsClient = boto3.client("s3", **s3_client_kwargs)
        # Create a Secrets Manager client
        self.awsSecretManagerClient = boto3.session.Session().client(**secrets_client_kwargs)

    def get_aws_secret_manager_value(self, key: str) -> dict:
        """fetch KAFKA AWS secret key from secret manager
        Args:
            key (string): secret manager key

        Raises:
            e: if the secret fetching failed

        Returns:
            dict: secret
        """
        if self.credentialViaAwsKey:
            try:
                getSecretValueResponse = self.awsSecretManagerClient.get_secret_value(
                    SecretId=key
                )
            except ClientError as e:
                # For a list of exceptions thrown, see
                # https://docs.aws.amazon.com/secretsmanager/latest/apireference/API_GetSecretValue.html
                raise e

            # Decrypts secret using the associated KMS key.
            secret = getSecretValueResponse["SecretString"]
            secret = eval(secret)
            return secret
        else:
            # If the secret is in the environment variables
            secret = deepcopy(self.envVariables)
            if key in self.secretMap.keys():
                secretFiltered = {
                    secretKey: secret[secretKey] for secretKey in self.secretMap[key]
                }
                secret = secretFiltered

            for key, secretVal in deepcopy(secret).items():
                if isinstance(key, str) and key.endswith("_pluang.com"):
                    # Replace the key suffix from _pluang.com to @pluang.com
                    # This is to ensure that the key is in the format of an email address
                    secret.pop(key)
                    secret[key.replace("_pluang.com", "@pluang.com")] = secretVal
            return secret

    def secret_manager_to_json(self, keys: list, filePath: str):
        """fetch KAFKA AWS secret keys from secret manager and store into a json file

        Args:
            keys (list): list of secret keys
            filePath (str): filepath to store the fetched secret values
        """
        d = {}
        for key in keys:
            secret = self.get_aws_secret_manager_value(key)
            d.update(secret)
        if d != {}:
            json_exporter(d, filePath)

    def upload_to_s3(self, local_file_path: str, s3_path: str) -> bool:
        """
        Upload a file to S3.
        
        Args:
            local_file_path (str): Local file path to upload
            s3_path (str): S3 path in format 's3://bucket-name/key/path' or 'bucket-name/key/path'
            
        Returns:
            bool: True if upload successful, False otherwise
        """
        try:
            # Parse S3 path
            if s3_path.startswith('s3://'):
                s3_path = s3_path[5:]  # Remove 's3://' prefix
            
            # Split bucket and key
            parts = s3_path.split('/', 1)
            bucket_name = parts[0]
            s3_key = parts[1] if len(parts) > 1 else ''
            
            # Upload file
            self.awsClient.upload_file(local_file_path, bucket_name, s3_key)
            return True
        except Exception as e:
            print(f"Error uploading to S3: {e}")
            return False

    def download_from_s3(self, s3_path: str, local_file_path: str) -> bool:
        """
        Download a file from S3.
        
        Args:
            s3_path (str): S3 path in format 's3://bucket-name/key/path' or 'bucket-name/key/path'
            local_file_path (str): Local file path to save the downloaded file
            
        Returns:
            bool: True if download successful, False otherwise
        """
        try:
            # Parse S3 path
            if s3_path.startswith('s3://'):
                s3_path = s3_path[5:]  # Remove 's3://' prefix
            
            # Split bucket and key
            parts = s3_path.split('/', 1)
            bucket_name = parts[0]
            s3_key = parts[1] if len(parts) > 1 else ''
            
            # Download file
            self.awsClient.download_file(bucket_name, s3_key, local_file_path)
            return True
        except ClientError as e:
            # Check if file doesn't exist (404)
            error_code = e.response.get('Error', {}).get('Code', '')
            if error_code == '404' or error_code == 'NoSuchKey':
                print(f"File not found in S3: {s3_path}")
            else:
                print(f"Error downloading from S3: {e}")
            return False
        except Exception as e:
            print(f"Error downloading from S3: {e}")
            return False

    def centralizedAlert(
        self,
        slackChannelName,
        title,
        assetClass,
        attentionEmailList,
        otherMsgs={},
        severity="info",
    ):
        centralizedSlackAlert = CentralizedSlackAlert(
            self.slackChannelNameUrlDict,
            self.slackIdDict,
            env=self.env,
            cloud=self.cloud,
        )

        centralizedSlackAlert.send_slack_msg(
            slackChannelName, title, assetClass, attentionEmailList, otherMsgs, severity
        )
