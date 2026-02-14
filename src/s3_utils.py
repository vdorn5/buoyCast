# s3_utils.py

from io import BytesIO
import boto3
from botocore.client import ClientError
import pandas as pd
import sagemaker


def get_s3_client(region_name: str = None) -> boto3.client:
    """
    Create an S3 client for the given region.

    Args:
        region_name (str, optional): AWS region name. Uses default if None.

    Returns:
        boto3.client: S3 client instance.
    """
    session = boto3.session.Session()
    if region_name is None:
        region_name = session.region_name

    s3_client = boto3.client("s3", region_name=region_name)
    return s3_client


def get_sagemaker_bucket(sagemaker_session: sagemaker.Session = None) -> str:
    """
    Retrieve the default SageMaker bucket. Creates one if it doesn't exist.

    Args:
        sagemaker_session (sagemaker.Session, optional): SageMaker session. Creates default if None.

    Returns:
        str: S3 bucket name.
    """
    if sagemaker_session is None:
        sagemaker_session = sagemaker.Session()

    bucket = sagemaker_session.default_bucket()
    return bucket


def verify_bucket(bucket: str, s3_client: boto3.client = None) -> bool:
    """
    Verify that an S3 bucket exists.

    Args:
        bucket (str): Name of the bucket.
        s3_client (boto3.client, optional): S3 client instance. Creates one if None.

    Returns:
        bool: True if bucket exists, False otherwise.
    """
    if s3_client is None:
        s3_client = get_s3_client()

    try:
        s3_client.head_bucket(Bucket=bucket)
        print(f"[INFO] Bucket '{bucket}' exists.")
        return True
    except ClientError as e:
        print(f"[ERROR] Cannot find bucket '{bucket}': {e}")
        return False


def upload_df_to_s3(
    df: pd.DataFrame,
    bucket: str,
    s3_prefix: str,
    buoy_id: str,
    file_name: str = "stdmet.parquet",
    s3_client: boto3.client = None
) -> str:
    """
    Upload a DataFrame as a Parquet file to S3.

    Args:
        df (pd.DataFrame): DataFrame to upload.
        bucket (str): S3 bucket name.
        s3_prefix (str): Prefix/path in the bucket.
        buoy_id (str): Buoy identifier for folder structure.
        file_name (str, optional): Parquet file name. Defaults to "stdmet.parquet".
        s3_client (boto3.client, optional): S3 client instance. Creates one if None.

    Returns:
        str: Full S3 path to the uploaded file.
    """
    if s3_client is None:
        s3_client = get_s3_client()

    buffer = BytesIO()
    df.to_parquet(buffer, index=False)
    buffer.seek(0)

    s3_key = f"{s3_prefix}/buoy={buoy_id}/{file_name}"

    s3_client.put_object(
        Bucket=bucket,
        Key=s3_key,
        Body=buffer.getvalue()
    )

    s3_path = f"s3://{bucket}/{s3_key}"
    print(f"[INFO] Uploaded to {s3_path}")
    return s3_path
