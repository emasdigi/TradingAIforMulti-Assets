import boto3
from botocore.exceptions import ClientError
from copy import deepcopy
import ast
import os

from utils.alertSys import CentralizedSlackAlert
from utils.utils import json_parser, json_exporter, yaml_parser


class AWS:
    def __init__(self, env: str, configFilePath: str):

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
        self.cloud = "AWS"
        self.credentialViaAwsKey = env in ["VM", "development"]

        # Load secret map only if not using AWS key credentials
        # For VM/development, we use AWS credentials instead
        if not self.credentialViaAwsKey:
            secret_map_path = f"/workspace/config/secret-map/{env}.json"
            if os.path.exists(secret_map_path):
                self.secretMap = json_parser(secret_map_path)
            else:
                self.secretMap = {}
        else:
            self.secretMap = {}

        if self.credentialViaAwsKey:
            self.aws_client_init()
        else:
            self.load_env_as_dict()
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

    def aws_client_init(self):
        """
        Initializes the AWS S3 client using credentials from a specified JSON file. This function
        extracts the necessary access key and secret key from the file, and sets up the S3 client
        using the `boto3` library. The JSON file path is obtained from `CONFIG["filePathAwsKey"]`.

        Raises:
            KeyError: If required credentials are not found in the JSON file.
        """
        # self.awsKey = json_parser(self.CONFIG["filePathAwsKey"])
        self.awsKey = {}
        # initiate AWS client, to read object from s3
        self.storageOptions = {}
        for key in ["aws_access_key_id", "access_key_ID"]:
            if key in self.awsKey.keys():
                self.storageOptions["key"] = self.awsKey[key]
                break
        for secret in ["aws_secret_access_key", "secret_access_key"]:
            if secret in self.awsKey.keys():
                self.storageOptions["secret"] = self.awsKey[secret]
                break
        self.awsClient = boto3.client("s3")
        # Create a Secrets Manager client
        self.awsSecretManagerClient = boto3.session.Session().client(
            service_name="secretsmanager",
            region_name=self.CONFIG["awsRegionName"],
        )

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
