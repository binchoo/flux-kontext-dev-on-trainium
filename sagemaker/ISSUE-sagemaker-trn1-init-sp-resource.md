# [Inference][SageMaker][trn1.32xlarge] TP optimum-neuron model fails at `init_sp_resource` (NRT_RESOURCE) on a real-time endpoint, while the identical image loads and infers via local `docker run --device`

[Summary]
Related to #1338 but on a SageMaker real-time endpoint instead of EKS. A TP8 `optimum-neuron` model (FLUX.1-Kontext, `NeuronFluxKontextPipeline`, 1024x1024, bf16) loads and serves under local `docker run --device`, but the byte-identical image fails to load on a SageMaker real-time endpoint on the same trn1.32xlarge host, with a single torchserve worker.

[Setup]
- Image built from the `pytorch-inference-neuronx` DLC `2.7.0-neuronx-py310-sdk2.25.0-ubuntu22.04`
- `optimum-neuron==0.4.5` and `diffusers==0.35.2`. 
- The `.neff` was recompiled inside this same image so the runtime matches the compiler (neuronx-cc 2.21, torch-neuronx 2.8, libneuronxla 2.2, NRT 2.27.23, driver 2.27.4).

[Works — local docker on the trn1 host]
```
docker run --rm -p 8080:8080 \
  --device /dev/neuron0 ... --device /dev/neuron7 \
  -e NEURON_RT_NUM_CORES=8 -e SAGEMAKER_MODEL_SERVER_WORKERS=1 \
  <image> serve

Default workers per model: 1
[model_fn] pipeline loaded + device init in 67.8s
POST /invocations 200, inference completed in 21.1s
```

[Fails — SageMaker real-time endpoint, same image, ml.trn1.32xlarge]
Torchserve reports `Default workers per model: 1` (only W-9000), but the single worker fails to stage the graph on every core context and retries indefinitely, returning 500 on every `/invocations`:
```
ERROR ENCD:init_sp_resource [nec_dev 4] tdrv_get_device_resource_va failed (ret=4) to get event semaphore vaddr mla->mla_idx=2 tpb->idx=0 spe->sp->idx=0
ERROR ENCD:encd_init_context [nec_dev 4] failed to init TOP_SP resources
ERROR TDRV:build_enc_ctx [nec_dev 4] failed to init context
ERROR NMGR:kmgr_load_nn_post_metrics Failed to load NN: /tmp/nxd_model/BaseModelInstance/_tp0_bk0/model.MODULE_....neff, err: 4
ERROR NRT:nrt_infodump Failure: NRT_RESOURCE in nrt_load_util
ERROR NRT:nrt_infodump Visible cores: 0, 1, 2, 3, 4, 5, 6, 7
RuntimeError: Could not load the model status=4 message=Allocation Failure
Backend worker process died. ... RuntimeError: 500 - Unknown exception
Retry worker: 9000 in 1 seconds.
```

[Ruled out]
- Worker count: `Default workers per model: 1` in both; the same single worker works locally.
- `SAGEMAKER_MODEL_SERVER_WORKERS=1`, `NEURON_RT_VISIBLE_CORES=0-7`, and `code/config.properties` (`default_workers_per_model=1`) — no change on the endpoint.
- Identical container image loads and infers locally.
- Model code `model_fn` succeeds locally.

The only remaining difference is how SageMaker hosting exposes NeuronCores to the container vs local `docker run --device /dev/neuron*`. The `tdrv_get_device_resource_va ... event semaphore vaddr` failure suggests the container cannot map the per-core event-semaphore region under SageMaker's device passthrough for a TP model on trn1.

`[Expected]`
When we set the worker count (`SAGEMAKER_MODEL_SERVER_WORKERS` / `default_workers_per_model`) and the core count (`NEURON_RT_NUM_CORES` / `NEURON_RT_VISIBLE_CORES`), the endpoint should bootstrap exactly that many model replicas over exactly those cores — e.g. one worker holding the 8 TP cores — the same way local `docker run --device` does. Instead the endpoint ignores those settings at the device level: a single worker still tries to bring up all core contexts and fails on each, looping forever.

`[Environment]`
- DLC base: pytorch-inference-neuronx:2.7.0-neuronx-py310-sdk2.25.0-ubuntu22.04
- optimum-neuron 0.4.5, diffusers 0.35.2, torch-neuronx 2.8, neuronx-cc 2.21, libneuronxla 2.2, NRT 2.27.23, driver 2.27.4
- ml.trn1.32xlarge, SageMaker real-time endpoint
- FLUX.1-Kontext, TP8, 1024x1024, bf16

> Note: redact the `Instance ID: i-...` line from any attached logs before posting.
