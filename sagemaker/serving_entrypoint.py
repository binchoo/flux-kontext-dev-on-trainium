#!/usr/bin/env python
"""Container entrypoint for the self-contained Neuron serving image (no DLC).

SageMaker starts the inference container with the argument `serve`. This boots
the `sagemaker-inference` model server, which implements the SageMaker hosting
contract and dispatches to the user handler at /opt/ml/model/code/inference.py
(model_fn / input_fn / predict_fn / output_fn) — exactly what the AWS DLC did.

Any other argument is exec'd verbatim (so `docker run ... bash` still works for
debugging).
"""
import subprocess
import sys


def _serve():
    # The sagemaker-inference model server reads SAGEMAKER_PROGRAM /
    # SAGEMAKER_SUBMIT_DIRECTORY (set in deploy_endpoint.py env) to locate the
    # handler module under /opt/ml/model/code, installs code/requirements.txt if
    # present, then serves model_fn/input_fn/predict_fn/output_fn over /invocations
    # and answers /ping for the health check.
    from sagemaker_inference import model_server

    # Our custom 3-component wiring (handler_service.py) bridges the model server
    # to the user's code/inference.py four functions. Reference it by module:class.
    model_server.start_model_server(handler_service="handler_service:HandlerService")


def main():
    if len(sys.argv) >= 2 and sys.argv[1] == "serve":
        _serve()
    else:
        # passthrough (e.g. `docker run ... python -c ...` or `bash`)
        sys.exit(subprocess.call(sys.argv[1:]))


if __name__ == "__main__":
    main()
