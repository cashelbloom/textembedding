import json
import boto3
import base64
from botocore.exceptions import ClientError


def get_secret():
    # Create a Secrets Manager client
    client = boto3.client("secretsmanager", region_name="us-east-1")
    secret_name = ""
    try:
        # Retrieve the secret
        get_secret_value_response = client.get_secret_value(SecretId=secret_name)
    except ClientError as e:
        # Handle errors
        if e.response["Error"]["Code"] == "DecryptionFailureException":
            raise e
        elif e.response["Error"]["Code"] == "InternalServiceErrorException":
            raise e
        elif e.response["Error"]["Code"] == "InvalidParameterException":
            raise e
        elif e.response["Error"]["Code"] == "InvalidRequestException":
            raise e
        elif e.response["Error"]["Code"] == "ResourceNotFoundException":
            raise e
    else:
        # Decrypts secret using the associated KMS CMK
        if "SecretString" in get_secret_value_response:
            secret = get_secret_value_response["SecretString"]
        else:
            secret = base64.b64decode(get_secret_value_response["SecretBinary"])
    print(f"data type of secret is: {type(secret)}")
    return json.loads(secret)["OPENAI_API_KEY"]
