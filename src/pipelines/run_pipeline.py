"""
CLI helper to upsert and start the buoyCast SageMaker Pipeline.

Usage (from repo root in SageMaker Studio):
  python -m pipelines.run_pipeline --region us-east-1 --role-arn <ROLE_ARN> \
      --curated-s3-prefix s3://<bucket>/curated/ndbc/ \
      --artifacts-prefix s3://<bucket>/buoycast/artifacts \
      --model-package-group buoycast-wave-models
"""

from __future__ import annotations

import argparse
import time

import boto3
import sagemaker

from pipelines.buoycast_pipeline import get_pipeline


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--region", type=str, default=boto3.Session().region_name)
    parser.add_argument("--role-arn", type=str, required=True)
    parser.add_argument("--pipeline-name", type=str, default="buoycast-train-register")
    parser.add_argument("--curated-s3-prefix", type=str, required=True)
    parser.add_argument("--artifacts-prefix", type=str, required=True)
    parser.add_argument("--model-package-group", type=str, default="buoycast-wave-models")
    parser.add_argument("--rmse-hs-threshold", type=float, default=0.5)
    parser.add_argument("--run-id", type=str, default=time.strftime("%Y%m%d-%H%M%S"))

    args = parser.parse_args()

    sess = sagemaker.Session(boto3.Session(region_name=args.region))
    bucket = sess.default_bucket()

    pipeline = get_pipeline(
        region=args.region,
        role=args.role_arn,
        default_bucket=bucket,
        pipeline_name=args.pipeline_name,
    )

    pipeline.upsert(role_arn=args.role_arn)

    execution = pipeline.start(
        parameters={
            "CuratedS3Prefix": args.curated_s3_prefix,
            "ArtifactsS3Prefix": args.artifacts_prefix,
            "ModelPackageGroupName": args.model_package_group,
            "RmseHsThreshold": args.rmse_hs_threshold,
            "RunId": args.run_id,
        }
    )

    print("Started execution:", execution.arn)


if __name__ == "__main__":
    main()
