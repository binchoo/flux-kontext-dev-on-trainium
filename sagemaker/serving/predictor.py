#!/usr/bin/env python
"""Flask inference app — SageMaker BYOC contract (/ping + /invocations).

No JVM: this replaces multi-model-server / torchserve (which have a Java Netty
frontend) with the AWS-documented Flask + Gunicorn + Nginx pattern. SageMaker
only requires a web server on :8080 answering /ping (GET) and /invocations
(POST) within 60s — it does NOT require Java.

The model itself is served by the SAME four functions we already wrote in
code/inference.py (model_fn / input_fn / predict_fn / output_fn). We import that
module from /opt/ml/model/code (where SageMaker extracts model.tar.gz's code/)
and call them. So the handler logic — including the Neuron pipeline load and the
letterbox aspect-ratio handling — is unchanged; only the serving shell differs.
"""
import importlib.util
import logging
import os
import sys

import flask

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _load_handler():
    """Import code/inference.py (the four-function handler) once."""
    code_dir = os.environ.get("SAGEMAKER_SUBMIT_DIRECTORY", "/opt/ml/model/code")
    program = os.environ.get("SAGEMAKER_PROGRAM", "inference.py")
    if code_dir not in sys.path:
        sys.path.insert(0, code_dir)
    path = os.path.join(code_dir, program)
    spec = importlib.util.spec_from_file_location("user_inference", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    logger.info("loaded handler from %s", path)
    return mod


class _ModelHolder:
    """Lazy singleton: load the Neuron pipeline once on first need (~50s)."""

    handler = None
    model = None

    @classmethod
    def get(cls):
        if cls.handler is None:
            cls.handler = _load_handler()
        if cls.model is None:
            model_dir = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")
            logger.info("calling model_fn(%s) ...", model_dir)
            cls.model = cls.handler.model_fn(model_dir)
            logger.info("model loaded")
        return cls.handler, cls.model


app = flask.Flask(__name__)


@app.route("/ping", methods=["GET"])
def ping():
    # Healthy once the handler module imports and the model has loaded. We force
    # the (slow) model load here so SageMaker only marks us InService when the
    # pipeline is actually ready to serve. The high container startup health
    # check timeout (deploy_endpoint.py) absorbs the ~50s Neuron init.
    try:
        _ModelHolder.get()
        return flask.Response(response="\n", status=200, mimetype="application/json")
    except Exception:  # noqa: BLE001
        logger.exception("ping failed: model not ready")
        return flask.Response(response="\n", status=404, mimetype="application/json")


@app.route("/invocations", methods=["POST"])
def invocations():
    handler, model = _ModelHolder.get()
    content_type = flask.request.content_type or "application/json"
    accept = flask.request.headers.get("Accept", "application/json")

    data = handler.input_fn(flask.request.get_data(), content_type)
    prediction = handler.predict_fn(data, model)
    body, out_ctype = handler.output_fn(prediction, accept)
    return flask.Response(response=body, status=200, mimetype=out_ctype)
