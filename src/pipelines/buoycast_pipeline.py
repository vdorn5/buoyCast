"""
SageMaker Pipeline definition for buoyCast (AAI-540).

Pipeline stages:
  1) Preprocess (Processing): read curated CSV -> supervised features -> train/val/test splits
  2) Train (Training): scikit-learn HistGradientBoostingRegressor
  3) Evaluate (Processing): compute regression metrics and write evaluation.json
  4) Condition + Register: if metric passes, register to Model Registry; else Fail

This produces a visible DAG in SageMaker Studio and supports "success" and "failed"
pipeline demo runs by changing the metric threshold.
"""

from __future__ import annotations

from pathlib import Path

import boto3
import sagemaker
#from sagemaker.inputs import TrainingInput
try:
    # Older SDKs
    from sagemaker.inputs import TrainingInput
except ModuleNotFoundError:
    # Newer SDKs
    from sagemaker.workflow.steps import TrainingInput

from sagemaker.model_metrics import MetricsSource, ModelMetrics
from sagemaker.sklearn.estimator import SKLearn
from sagemaker.sklearn.model import SKLearnModel
from sagemaker.workflow.conditions import ConditionLessThanOrEqualTo
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.execution_variables import ExecutionVariables
from sagemaker.workflow.fail_step import FailStep
from sagemaker.workflow.functions import Join, JsonGet
from sagemaker.workflow.parameters import ParameterFloat, ParameterString
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.steps import ProcessingStep, TrainingStep
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.sklearn.processing import SKLearnProcessor
#from sagemaker.workflow.model_step import RegisterModel
from sagemaker.workflow.steps import CacheConfig
from sagemaker.workflow.parameters import ParameterString
from sagemaker.workflow.parameters import ParameterString, ParameterInteger, ParameterFloat


def get_pipeline(
    region: str,
    role: str,
    default_bucket: str | None = None,
    pipeline_name: str = "buoycast-train-register",
    base_job_prefix: str = "buoycast",
) -> Pipeline:
    """Create a SageMaker Pipeline object (does not upsert/run)."""

    boto_session = boto3.Session(region_name=region)
    sm_session = sagemaker.Session(boto_session=boto_session)
    default_bucket = default_bucket or sm_session.default_bucket()

    pipeline_session = PipelineSession(
        boto_session=boto_session,
        sagemaker_client=sm_session.sagemaker_client,
        default_bucket=default_bucket,
    )

    repo_root = Path(__file__).resolve().parents[1]
    scripts_dir = repo_root / "sagemaker_scripts"

    # -------------------------
    # Parameters
    # -------------------------
    curated_s3_prefix = ParameterString(
        name="CuratedS3Prefix",
        default_value=f"s3://{default_bucket}/curated/ndbc/",
    )

    artifacts_prefix = ParameterString(
        name="ArtifactsS3Prefix",
        default_value=f"s3://{default_bucket}/buoycast/artifacts",
    )

    # Use execution id by default, but allow manual override for repeatable demos.
    #run_id = ParameterString(
    #    name="RunId",
    #    default_value=ExecutionVariables.PIPELINE_EXECUTION_ID,
    #)
    run_id = ParameterString(
    name="RunId",
    default_value="auto",
    )

    model_package_group_name = ParameterString(
        name="ModelPackageGroupName",
        default_value="buoycast-wave-models",
    )

    model_approval_status = ParameterString(
        name="ModelApprovalStatus",
        default_value="Approved",  # set to PendingManualApproval if you want a manual gate
    )

    metric_threshold = ParameterFloat(
        name="RmseHsThreshold",
        default_value=0.5,
    )

    processing_instance_type = ParameterString(
        name="ProcessingInstanceType",
        default_value="ml.m5.xlarge",
    )

    training_instance_type = ParameterString(
        name="TrainingInstanceType",
        default_value="ml.m5.xlarge",
    )

    cache_config = CacheConfig(enable_caching=False)

    # -------------------------
    # Preprocess step
    # -------------------------
    processor = SKLearnProcessor(
        framework_version="1.2-1",
        role=role,
        instance_type=processing_instance_type,
        instance_count=1,
        base_job_name=f"{base_job_prefix}/preprocess",
        sagemaker_session=pipeline_session,
    )

    train_dest = Join(on="/", values=[artifacts_prefix, run_id, "data", "train"])
    val_dest = Join(on="/", values=[artifacts_prefix, run_id, "data", "validation"])
    test_dest = Join(on="/", values=[artifacts_prefix, run_id, "data", "test"])
    baseline_dest = Join(on="/", values=[artifacts_prefix, run_id, "data", "baseline"])
    report_dest = Join(on="/", values=[artifacts_prefix, run_id, "reports", "data"])

    step_preprocess = ProcessingStep(
        name="Preprocess",
        processor=processor,
        inputs=[
            ProcessingInput(source=curated_s3_prefix, destination="/opt/ml/processing/input/curated"),
        ],
        outputs=[
            ProcessingOutput(output_name="train", source="/opt/ml/processing/train", destination=train_dest),
            ProcessingOutput(output_name="validation", source="/opt/ml/processing/validation", destination=val_dest),
            ProcessingOutput(output_name="test", source="/opt/ml/processing/test", destination=test_dest),
            ProcessingOutput(output_name="baseline", source="/opt/ml/processing/baseline", destination=baseline_dest),
            ProcessingOutput(output_name="report", source="/opt/ml/processing/report", destination=report_dest),
        ],
        code=str(scripts_dir / "preprocess.py"),
        cache_config=cache_config,
    )

    # -------------------------
    # Training step
    # -------------------------
    estimator = SKLearn(
        entry_point="train.py",
        source_dir=str(scripts_dir),
        framework_version="1.2-1",
        py_version="py3",
        role=role,
        instance_type=training_instance_type,
        instance_count=1,
        sagemaker_session=pipeline_session,
        hyperparameters={
            "max_iter": 400,
            "learning_rate": 0.05,
            "max_depth": 6,
            "min_samples_leaf": 30,
            "random_state": 42,
        },
    )

    step_train = TrainingStep(
        name="Train",
        estimator=estimator,
        inputs={
            "train": TrainingInput(
                s3_data=step_preprocess.properties.ProcessingOutputConfig.Outputs["train"].S3Output.S3Uri,
                content_type="text/csv",
            ),
            "validation": TrainingInput(
                s3_data=step_preprocess.properties.ProcessingOutputConfig.Outputs["validation"].S3Output.S3Uri,
                content_type="text/csv",
            ),
        },
        cache_config=cache_config,
    )

    # -------------------------
    # Evaluation step
    # -------------------------
    eval_processor = SKLearnProcessor(
        framework_version="1.2-1",
        role=role,
        instance_type=processing_instance_type,
        instance_count=1,
        base_job_name=f"{base_job_prefix}/evaluate",
        sagemaker_session=pipeline_session,
    )

    eval_dest = Join(on="/", values=[artifacts_prefix, run_id, "reports", "evaluation"])

    evaluation_report = PropertyFile(
        name="EvaluationReport",
        output_name="evaluation",
        path="evaluation.json",
    )

    step_eval = ProcessingStep(
        name="Evaluate",
        processor=eval_processor,
        inputs=[
            ProcessingInput(
                source=step_train.properties.ModelArtifacts.S3ModelArtifacts,
                destination="/opt/ml/processing/model",
            ),
            ProcessingInput(
                source=step_preprocess.properties.ProcessingOutputConfig.Outputs["test"].S3Output.S3Uri,
                destination="/opt/ml/processing/test",
            ),
        ],
        outputs=[
            ProcessingOutput(
                output_name="evaluation",
                source="/opt/ml/processing/evaluation",
                destination=eval_dest,
            ),
        ],
        code=str(scripts_dir / "evaluate.py"),
        property_files=[evaluation_report],
        cache_config=cache_config,
    )

    # -------------------------
    # Condition + Register
    # -------------------------
    rmse_hs = JsonGet(
        step_name=step_eval.name,
        property_file=evaluation_report,
        json_path="regression_metrics.rmse_Hs",
    )

    model_metrics = ModelMetrics(
        model_statistics=MetricsSource(
            s3_uri=Join(
                on="/",
                values=[
                    step_eval.properties.ProcessingOutputConfig.Outputs["evaluation"].S3Output.S3Uri,
                    "evaluation.json",
                ],
            ),
            content_type="application/json",
        )
    )

    model = SKLearnModel(
        model_data=step_train.properties.ModelArtifacts.S3ModelArtifacts,
        role=role,
        entry_point="inference.py",
        source_dir=str(scripts_dir),
        framework_version="1.2-1",
        py_version="py3",
        sagemaker_session=pipeline_session,
    )

    #step_register = RegisterModel(
    #    name="RegisterModel",
    #    model=model,
    #    model_package_group_name=model_package_group_name,
    #    model_metrics=model_metrics,
    #    approval_status=model_approval_status,
    #    content_types=["text/csv", "application/json"],
    #    response_types=["application/json", "text/csv"],
    #    inference_instances=["ml.m5.large", "ml.m5.xlarge"],
    #    transform_instances=["ml.m5.large"],
    #)

    step_fail = FailStep(
        name="FailIfBadMetrics",
        error_message=Join(
            on="",
            values=["RMSE(Hs) did not meet threshold. Got: ", rmse_hs],
        ),
    )

    step_condition = ConditionStep(
        name="CheckMetrics",
        conditions=[
            ConditionLessThanOrEqualTo(left=rmse_hs, right=metric_threshold),
        ],
        #if_steps=[step_register]
        if_steps=[],
        else_steps=[step_fail],
    )

    pipeline = Pipeline(
        name=pipeline_name,
        parameters=[
            curated_s3_prefix,
            artifacts_prefix,
            run_id,
            model_package_group_name,
            model_approval_status,
            metric_threshold,
            processing_instance_type,
            training_instance_type,
        ],
        steps=[step_preprocess, step_train, step_eval, step_condition],
        sagemaker_session=pipeline_session,
    )

    return pipeline
