import importlib
import os
import sys

from loguru import logger

logger.remove()
logger.add(sys.stdout, level="WARNING")

AVAILABLE_MODELS = {
    "qwen2vl": "Qwen2VL",
    "gemini": "GeminiModels",
    "openai": "OpenaiModels",
    "lmdeploy_models": "LMDeployModels",
    "vllm_models": "VllmModels",
}


def get_model(model_name):
    if model_name not in AVAILABLE_MODELS:
        raise ValueError(f"Model {model_name} not found in available models.")

    model_class = AVAILABLE_MODELS[model_name]
    try:
        module = __import__(f"tool_server.tf_eval.models.{model_name}", fromlist=[model_class])
        return getattr(module, model_class)
    except Exception as e:
        logger.error(f"Failed to import {model_class} from {model_name}: {e}")
        raise
