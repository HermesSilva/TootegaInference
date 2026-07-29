# Cosmologia fine-tune -- remote environment.
#
# Source this before anything:  . /workspace/cosmologia/env.sh
#
# Two of these are not cosmetic.  The container root is a 20 GB overlay, so
# HF_HOME must not fall back to ~/.cache -- a 62 GB checkpoint would fill the
# filesystem and take the box down with it.  And expandable_segments keeps the
# allocator from fragmenting around 62 GB of resident weights, which is what
# turns a run that fits on paper into an OOM at step 40.

export COSMO_ROOT=/workspace/cosmologia

export HF_HOME=$COSMO_ROOT/hf
export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_VERBOSITY=warning

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_DEVICE_MAX_CONNECTIONS=1

export PATH=$COSMO_ROOT/venv/bin:$PATH

# The box is shared with another project (LTX-2, PildraEngine).  Everything this
# work writes stays under $COSMO_ROOT.
