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

    def upload_directory_to_s3(self, local_directory: str, s3_base_path: str, pattern: str = "*") -> dict:
        """
        Upload all files in a directory to S3.
        
        Args:
            local_directory (str): Local directory path containing files to upload
            s3_base_path (str): S3 base path in format 's3://bucket-name/key/path/' or 'bucket-name/key/path/'
            pattern (str): Glob pattern to filter files (default: "*" for all files)
            
        Returns:
            dict: Summary with 'success' count, 'failed' count, and 'uploaded_files' list
        """
        import glob
        from pathlib import Path
        
        # Ensure s3_base_path ends with '/'
        if not s3_base_path.endswith('/'):
            s3_base_path += '/'
        
        # Get all matching files
        local_dir_path = Path(local_directory)
        if not local_dir_path.exists():
            print(f"Directory not found: {local_directory}")
            return {"success": 0, "failed": 0, "uploaded_files": []}
        
        # Use glob to find matching files
        files = list(local_dir_path.glob(pattern))
        
        success_count = 0
        failed_count = 0
        uploaded_files = []
        
        for file_path in files:
            if file_path.is_file():  # Only process files, not directories
                # Get the relative filename
                filename = file_path.name
                
                # Construct S3 path
                s3_file_path = s3_base_path + filename
                
                # Upload file
                success = self.upload_to_s3(str(file_path), s3_file_path)
                
                if success:
                    success_count += 1
                    uploaded_files.append(filename)
                    print(f"✓ Uploaded: {filename} → {s3_file_path}")
                else:
                    failed_count += 1
                    print(f"✗ Failed to upload: {filename}")
        
        result = {
            "success": success_count,
            "failed": failed_count,
            "uploaded_files": uploaded_files
        }
        
        print(f"\nUpload Summary: {success_count} succeeded, {failed_count} failed")
        return result

    def download_directory_from_s3(self, s3_base_path: str, local_directory: str, pattern: str = "*") -> dict:
        """
        Download all files from an S3 path to a local directory.
        
        Args:
            s3_base_path (str): S3 base path in format 's3://bucket-name/key/path/' or 'bucket-name/key/path/'
            local_directory (str): Local directory path to save downloaded files
            pattern (str): Pattern to filter files (e.g., "*.csv", "*.json", default: "*" for all files)
            
        Returns:
            dict: Summary with 'success' count, 'failed' count, and 'downloaded_files' list
        """
        from pathlib import Path
        
        try:
            # Parse S3 path
            if s3_base_path.startswith('s3://'):
                s3_base_path_cleaned = s3_base_path[5:]
            else:
                s3_base_path_cleaned = s3_base_path
            
            # Split bucket and prefix
            parts = s3_base_path_cleaned.split('/', 1)
            bucket_name = parts[0]
            prefix = parts[1] if len(parts) > 1 else ''
            
            # Ensure prefix ends with '/' if it exists
            if prefix and not prefix.endswith('/'):
                prefix += '/'
            
            # Create local directory if it doesn't exist
            local_dir_path = Path(local_directory)
            local_dir_path.mkdir(parents=True, exist_ok=True)
            
            # List objects in S3
            response = self.awsClient.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
            
            if 'Contents' not in response:
                print(f"No files found in S3 path: {s3_base_path}")
                return {"success": 0, "failed": 0, "downloaded_files": []}
            
            success_count = 0
            failed_count = 0
            downloaded_files = []
            
            # Process pattern matching
            import fnmatch
            
            for obj in response['Contents']:
                s3_key = obj['Key']
                
                # Skip if it's a directory marker
                if s3_key.endswith('/'):
                    continue
                
                # Get filename from key
                filename = s3_key.split('/')[-1]
                
                # Check if filename matches pattern
                if not fnmatch.fnmatch(filename, pattern):
                    continue
                
                # Construct local file path
                local_file_path = local_dir_path / filename
                
                # Construct full S3 path for download
                s3_file_path = f"s3://{bucket_name}/{s3_key}"
                
                # Download file
                success = self.download_from_s3(s3_file_path, str(local_file_path))
                
                if success:
                    success_count += 1
                    downloaded_files.append(filename)
                    print(f"✓ Downloaded: {s3_file_path} → {filename}")
                else:
                    failed_count += 1
                    print(f"✗ Failed to download: {filename}")
            
            result = {
                "success": success_count,
                "failed": failed_count,
                "downloaded_files": downloaded_files
            }
            
            print(f"\nDownload Summary: {success_count} succeeded, {failed_count} failed")
            return result
            
        except Exception as e:
            print(f"Error listing S3 objects: {e}")
            return {"success": 0, "failed": 0, "downloaded_files": []}

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
