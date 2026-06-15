#!/bin/bash
set -euo pipefail

WORKSPACE=/workspace
SETUP_FLAG=$WORKSPACE/.setup-done
STEPS=$WORKSPACE/.setup-steps
LOGS=$WORKSPACE/logs
mkdir -p "$LOGS" "$WORKSPACE/models" "$WORKSPACE/tmp" "$STEPS"
mkdir -p "$WORKSPACE/hf_cache" "$WORKSPACE/pip_cache" "$WORKSPACE/npm_cache"

export TMPDIR=$WORKSPACE/tmp
export HF_HOME=$WORKSPACE/hf_cache
export PIP_CACHE_DIR=$WORKSPACE/pip_cache
export NPM_CONFIG_CACHE=$WORKSPACE/npm_cache

# Model downloaded into $WORKSPACE/models on first boot (override via pod env).
# (EVA-Qwen2.5-72B was the prior default; Magnum is less "short-story narrator"
# and more conversational for RP. Set MODEL_REPO to swap back.)
MODEL_REPO=${MODEL_REPO:-BigHuggyD/TheDrummer_Behemoth-123B-v1_exl2_5.0bpw_h6}
MODEL_BRANCH=${MODEL_BRANCH:-main}
MODEL_NAME=$(basename "$MODEL_REPO")

# Draft model for speculative decoding. DRAFT_NAME is the local dir name and is
# the single source of truth shared by the download, the validator, and
# config.yml. It includes the branch so it documents the quant and so changing
# the branch points everything at a fresh dir.
DRAFT_REPO=${DRAFT_REPO:-bartowski/Mistral-7B-Instruct-v0.3-exl2}
DRAFT_BRANCH=${DRAFT_BRANCH:-6_5}
DRAFT_NAME=${DRAFT_NAME:-$(basename "$DRAFT_REPO")-$DRAFT_BRANCH}

ST_USERNAME=${ST_USERNAME:-admin}
ST_PASSWORD=${ST_PASSWORD:?Set ST_PASSWORD in the pod env vars}

# Run a named setup step only if its flag is absent; flag it on success.
# A failed step leaves no flag, so a re-run resumes from there instead of
# redoing the steps that already succeeded.
step() {
  local name=$1
  if [ -f "$STEPS/$name" ]; then
    # If a step_<name>_check function exists, run it to confirm the artifact
    # is still present. If the check fails, the flag is stale (artifact was
    # deleted, flag was touched prematurely, etc.) -- re-run the step instead
    # of trusting the flag blindly.
    if declare -F "step_${name}_check" >/dev/null && ! "step_${name}_check"; then
      echo "=== [stale] $name (flag set but artifact missing) -- re-running ==="
      rm -f "$STEPS/$name"
    else
      echo "=== [skip] $name ==="
      return 0
    fi
  fi
  echo "=== [run]  $name ==="
  "step_$name"
  touch "$STEPS/$name"
}

# Clone a git repo shallow (--depth=1) with up to 3 retries. MooseFS / network
# hiccups during clone surface as "inflate: data stream error" -- a clean retry
# usually fixes it. Removes any partial clone before each attempt.
git_clone_retry() {
  local url=$1
  local dest=$2
  shift 2
  local attempt
  for attempt in 1 2 3; do
    rm -rf "$dest"
    if git clone --depth=1 "$@" "$url" "$dest"; then
      return 0
    fi
    echo "    git clone $url failed (attempt $attempt/3), retrying in 5s..." >&2
    sleep 5
  done
  echo "ERROR: git clone $url failed after 3 attempts" >&2
  return 1
}

# Per-step validators: if any of these returns non-zero when its flag is set,
# `step` treats the flag as stale and re-runs the step. Recovers from someone
# deleting an artifact, or a flag being touched prematurely.
step_clone_tabby_check()        { [ -d "$WORKSPACE/tabbyAPI/.git" ]; }
step_venv_tabby_check()         { [ -x "$WORKSPACE/tabbyAPI/venv/bin/python" ]; }
step_config_tabby_check()       { [ -f "$WORKSPACE/tabbyAPI/config.yml" ]; }
step_download_model_check()     { [ -d "$WORKSPACE/models/$MODEL_NAME" ]; }
step_download_draft_check()     { [ -d "$WORKSPACE/models/$DRAFT_NAME" ]; }
step_clone_sillytavern_check()  { [ -d "$WORKSPACE/SillyTavern/.git" ]; }
step_npm_sillytavern_check()    { [ -d "$WORKSPACE/SillyTavern/node_modules" ]; }
step_config_sillytavern_check() { [ -f "$WORKSPACE/SillyTavern/config.yaml" ]; }

step_apt() {
  apt-get update -qq
  apt-get install -y -qq curl git build-essential ca-certificates nano
  curl -fsSL https://deb.nodesource.com/setup_20.x | bash - >/dev/null
  apt-get install -y -qq nodejs
}

step_clone_tabby() {
  cd "$WORKSPACE"
  if [ ! -d tabbyAPI/.git ]; then
    git_clone_retry https://github.com/theroyallab/tabbyAPI tabbyAPI
  fi
}

step_venv_tabby() {
  cd "$WORKSPACE/tabbyAPI"

  # The base image ships PyTorch in the system Python. Create the venv with
  # --system-site-packages so pip can reuse that torch instead of pulling the
  # multi-GB CUDA wheels again. If tabbyAPI pins a different torch, pip just
  # installs it into the venv (which shadows the system copy) -- so this is
  # never slower than an isolated venv, only sometimes much faster.
  local sys_torch venv_torch
  sys_torch=$(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo none)
  echo "    base image torch: $sys_torch"

  python -m venv venv --system-site-packages
  source venv/bin/activate
  pip install -U pip wheel
  pip install -U ".[cu12]"
  deactivate

  venv_torch=$(venv/bin/python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo none)
  if [ "$venv_torch" = "$sys_torch" ]; then
    echo "    torch: reused base image copy ($venv_torch) -- no reinstall"
  else
    echo "    torch: venv installed its own ($venv_torch); base image had $sys_torch"
  fi
}

step_config_tabby() {
  cat > "$WORKSPACE/tabbyAPI/config.yml" <<TABBYCFG
network:
  host: 0.0.0.0
  port: 5000
  disable_auth: false
  api_servers: ["OAI"]
logging:
  log_prompt: false
  log_generation_params: false
  log_requests: false
model:
  model_dir: $WORKSPACE/models
  model_name: $MODEL_NAME
  inline_model_loading: true
  max_seq_len: 32768
  cache_mode: Q8
  gpu_split_auto: true
  tensor_parallel: false   
  chunk_size: 4096
  draft_model:
    draft_model_dir: $WORKSPACE/models
    draft_model_name: $DRAFT_NAME
    draft_rope_scale: 1.0
    draft_cache_mode: FP16
developer:
  cuda_malloc_backend: true
TABBYCFG
}

step_download_model() {
  # exl2 weights for TabbyAPI live in a subdir of model_dir. A 72B ~4.5bpw
  # quant is ~43 GB -- make sure the volume has room. hf_transfer speeds it up.
  pip install -U "huggingface_hub[hf_transfer]"
  HF_HUB_ENABLE_HF_TRANSFER=1 hf download "$MODEL_REPO" \
    --revision "$MODEL_BRANCH" \
    --local-dir "$WORKSPACE/models/$MODEL_NAME"
}

step_download_draft() {
  # `hf` lives in the ephemeral system Python and doesn't persist across pods.
  # Install it here too so this step works even when download_model (which also
  # installs it) was skipped because the main model was already on the volume.
  pip install -U "huggingface_hub[hf_transfer]"
  HF_HUB_ENABLE_HF_TRANSFER=1 hf download "$DRAFT_REPO" \
    --revision "$DRAFT_BRANCH" \
    --local-dir "$WORKSPACE/models/$DRAFT_NAME"
}




step_clone_sillytavern() {
  cd "$WORKSPACE"
  if [ ! -d SillyTavern/.git ]; then
    git_clone_retry https://github.com/SillyTavern/SillyTavern SillyTavern -b staging
  fi
}

step_npm_sillytavern() {
  cd "$WORKSPACE/SillyTavern"
  npm install --no-audit --no-fund --loglevel=error
}

step_config_sillytavern() {
  cd "$WORKSPACE/SillyTavern"
  cp default/config.yaml config.yaml
  sed -i 's/^listen:.*/listen: true/' config.yaml
  sed -i 's/^port:.*/port: 8000/' config.yaml
  sed -i 's/^whitelistMode:.*/whitelistMode: false/' config.yaml
  sed -i 's/^basicAuthMode:.*/basicAuthMode: true/' config.yaml
  sed -i "s|^  username:.*|  username: \"$ST_USERNAME\"|" config.yaml
  sed -i "s|^  password:.*|  password: \"$ST_PASSWORD\"|" config.yaml
}

# apt packages install to the container root fs (/usr/bin, /var/lib/dpkg, ...),
# NOT /workspace -- they do not persist on a network volume. Run this every boot
# so git/node/etc. are present even when the heavy steps below are skipped.
echo "=== Installing system packages ==="
step_apt

if [ ! -f "$SETUP_FLAG" ]; then
  echo "=== First boot: installing dependencies (15-20 min) ==="
  echo "    ($SETUP_FLAG not found -- if you expected a cached setup,"
  echo "     your /workspace is NOT a reused network volume)"

  step clone_tabby
  step venv_tabby
  step config_tabby
  step download_model
  step download_draft
  step clone_sillytavern
  step npm_sillytavern
  step config_sillytavern

  touch "$SETUP_FLAG"
  echo "=== Setup complete ==="
else
  echo "=== Reusing persistent setup at $WORKSPACE (skipping install) ==="
fi

# Stop anything a previous run left behind so re-running boot.sh cleanly
# restarts both services instead of failing to bind ports 5000/8000.
echo "=== Stopping any existing TabbyAPI / SillyTavern ==="
pkill -f 'python -u main.py' 2>/dev/null || true
pkill -f 'node server.js'    2>/dev/null || true
sleep 2

echo "=== Starting TabbyAPI on :5000 ==="
cd "$WORKSPACE/tabbyAPI"
source venv/bin/activate
nohup python -u main.py > "$LOGS/tabby.log" 2>&1 &
TABBY_PID=$!

echo "=== Starting SillyTavern on :8000 ==="
cd "$WORKSPACE/SillyTavern"
nohup node server.js --listen --port 8000 > "$LOGS/sillytavern.log" 2>&1 &

# Wait for TabbyAPI to write api_tokens.yml so the banner can show the keys.
# Don't wait for "Application startup complete" -- with model_name set, that
# only fires after the (multi-minute) model load. The tail -f below streams
# the load progress live.
echo "=== Waiting for TabbyAPI to come up... ==="
for i in $(seq 1 120); do
  if [ -s "$WORKSPACE/tabbyAPI/api_tokens.yml" ]; then break; fi
  sleep 1
done

# Read keys from api_tokens.yml -- correct on first boot and on a reused volume
# (TabbyAPI only echoes the keys to the log on first generation).
POD_ID=${RUNPOD_POD_ID:-unknown}
TOKENS=$WORKSPACE/tabbyAPI/api_tokens.yml
API_KEY=$(grep -E '^api_key:'    "$TOKENS" 2>/dev/null | awk '{print $2}' || true)
ADMIN_KEY=$(grep -E '^admin_key:' "$TOKENS" 2>/dev/null | awk '{print $2}' || true)
: "${API_KEY:=(not found -- check $TOKENS)}"
: "${ADMIN_KEY:=(not found -- check $TOKENS)}"

cat <<BANNER

==========================================
  TabbyAPI:    https://${POD_ID}-5000.proxy.runpod.net
  SillyTavern: https://${POD_ID}-8000.proxy.runpod.net
  Tabby API key:   $API_KEY
  Tabby admin key: $ADMIN_KEY
  ST login:        $ST_USERNAME / (your ST_PASSWORD)
  Model:           $MODEL_NAME
                   (loading into VRAM -- watch the log below)
==========================================
BANNER

tail -f "$LOGS"/*.log
