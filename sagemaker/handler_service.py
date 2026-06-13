#!/usr/bin/env python
"""SageMaker handler service for the self-contained Neuron serving image.

The AWS DLC bundled `sagemaker_pytorch_serving_container`, which wires the
user's code/inference.py (model_fn/input_fn/predict_fn/output_fn) into the
sagemaker-inference model server. We have no DLC, so we implement that wiring
ourselves with the three required components (confirmed via the
sagemaker-inference-toolkit docs):

    DefaultInferenceHandler  -> delegates to the user's four functions
    Transformer              -> request/response flow around the handler
    DefaultHandlerService    -> what start_model_server(handler_service=...) loads

The user handler is located the same way the DLC did it: via the env vars
SAGEMAKER_SUBMIT_DIRECTORY (/opt/ml/model/code) + SAGEMAKER_PROGRAM
(inference.py), which deploy_endpoint.py sets.
"""
import importlib.util
import os
import sys

from sagemaker_inference import content_types, environment
from sagemaker_inference.default_handler_service import DefaultHandlerService
from sagemaker_inference.default_inference_handler import DefaultInferenceHandler
from sagemaker_inference.transformer import Transformer


def _load_user_module():
    """Import the user's inference.py from SAGEMAKER_SUBMIT_DIRECTORY.

    Mirrors the DLC behaviour: code dir is on sys.path and the program module
    (default inference.py) is imported so its model_fn/input_fn/predict_fn/
    output_fn are available.
    """
    code_dir = os.environ.get("SAGEMAKER_SUBMIT_DIRECTORY", environment.code_dir)
    program = os.environ.get("SAGEMAKER_PROGRAM", "inference.py")
    if code_dir and code_dir not in sys.path:
        sys.path.insert(0, code_dir)
    module_path = os.path.join(code_dir, program)
    spec = importlib.util.spec_from_file_location("user_inference", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class UserInferenceHandler(DefaultInferenceHandler):
    """Delegate each of the four contract functions to the user's inference.py.

    DefaultInferenceHandler provides fallbacks; we override all four to call the
    user functions when present (our inference.py defines all four).
    """

    def __init__(self):
        super().__init__()
        self._user = _load_user_module()

    def default_model_fn(self, model_dir, context=None):
        return self._user.model_fn(model_dir)

    def default_input_fn(self, input_data, content_type, context=None):
        return self._user.input_fn(input_data, content_type)

    def default_predict_fn(self, data, model, context=None):
        return self._user.predict_fn(data, model)

    def default_output_fn(self, prediction, accept, context=None):
        return self._user.output_fn(prediction, accept)


class HandlerService(DefaultHandlerService):
    """Loaded by start_model_server(handler_service="handler_service:HandlerService")."""

    def __init__(self):
        transformer = Transformer(default_inference_handler=UserInferenceHandler())
        super().__init__(transformer=transformer)
