"""shared_adapter_pool: train LoRA adapter pools and score them on SST2.

Migrated from the research repo ``glad`` (src/llm_pipeline + src/discoveries/
sst2_perf_prediction). Kept deliberately import-light at the top level so
``import shared_adapter_pool`` pulls in no torch / datasets / trl.
"""
