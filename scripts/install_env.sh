#!/usr/bin/env bash
# One-click environment installation for Awesome-Perception-Aware-RLVR (EasyR1-based).
#
# Quick start:
#   bash scripts/install_env.sh
#   ENV_NAME=my-env bash scripts/install_env.sh
#   INSTALL_QWEN35_FASTPATH=1 bash scripts/install_env.sh   # also build Qwen3.5 kernels
#   QWEN35_FASTPATH_ONLY=1 bash scripts/install_env.sh      # add them to an existing env
#
# By default this creates a CUDA 12.8 / vLLM 0.19 environment. Python packages are
# pinned through scripts/constraints.txt to the versions the repository was tested
# with. The optional Qwen3.5 fast path (INSTALL_QWEN35_FASTPATH=1) intentionally pins
# flash-linear-attention and builds causal-conv1d from source without deps so
# torch, transformers, and triton are not upgraded by transitive resolution.
# Run `scripts/install_env.sh --help` for all supported environment variables.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
TEMP_DIR="$(mktemp -d)"
trap 'rm -rf "${TEMP_DIR}"' EXIT

cd "${REPO_ROOT}"

ENV_NAME="${ENV_NAME:-parlvr}"
PYTHON_VERSION="3.11"
PYTHON_WHEEL_TAG="cp311-cp311"
CUDA_TOOLKIT_VERSION="${CUDA_TOOLKIT_VERSION:-12.8.1}"
CUDA_LABEL="${CUDA_LABEL:-cuda-12.8.1}"
VLLM_VERSION="${VLLM_VERSION:-0.19.0}"
FLASH_ATTN_VERSION="${FLASH_ATTN_VERSION:-2.8.3}"
INSTALL_QWEN35_FASTPATH="${INSTALL_QWEN35_FASTPATH:-0}"
QWEN35_FASTPATH_ONLY="${QWEN35_FASTPATH_ONLY:-0}"
if [[ "${QWEN35_FASTPATH_ONLY}" == "1" ]]; then
    INSTALL_QWEN35_FASTPATH=1
fi
FLA_VERSION="${FLA_VERSION:-0.4.2}"
CAUSAL_CONV1D_REF="${CAUSAL_CONV1D_REF:-v1.6.2.post1}"
INSTALL_CONDA_CUDA="${INSTALL_CONDA_CUDA:-1}"
INSTALL_DEV_TOOLS="${INSTALL_DEV_TOOLS:-1}"
CPP_RUNTIME_CHANNEL="${CPP_RUNTIME_CHANNEL:-conda-forge}"
CPP_RUNTIME_REFRESHED=0
RUNTIME_HOOKS_INSTALLED=0
NEEDS_REACTIVATE_NOTICE=0

usage() {
    cat <<EOF
Awesome-Perception-Aware-RLVR environment installer.

Usage:
  scripts/install_env.sh
  INSTALL_QWEN35_FASTPATH=1 scripts/install_env.sh
  QWEN35_FASTPATH_ONLY=1 ENV_NAME=parlvr scripts/install_env.sh
  scripts/install_env.sh --help

What it installs:
  - Python ${PYTHON_VERSION} conda environment.
  - CUDA toolkit ${CUDA_TOOLKIT_VERSION} inside the conda environment by default.
  - vLLM ${VLLM_VERSION} with the cu128 torch backend.
  - Runtime dependencies from requirements.txt, excluding vllm and flash-attn.
  - flash-attn ${FLASH_ATTN_VERSION}, with conda C++ runtime hooks if needed.
  - Qwen3.5 fast-path dependencies, only with INSTALL_QWEN35_FASTPATH=1 (off by
    default; only Qwen3.5 models use them, other models do not need them):
      fla-core==${FLA_VERSION}
      flash-linear-attention==${FLA_VERSION}
      causal-conv1d built from ${CAUSAL_CONV1D_REF} source with --no-deps
  - EasyR1 in editable mode.

Common examples:
  # Default environment, without the Qwen3.5 fast path.
  scripts/install_env.sh

  # Default environment plus the Qwen3.5 fast path.
  INSTALL_QWEN35_FASTPATH=1 scripts/install_env.sh

  # Add the Qwen3.5 fast path to an environment installed earlier, changing nothing
  # else; skipped when it already works.
  QWEN35_FASTPATH_ONLY=1 ENV_NAME=parlvr scripts/install_env.sh

  # Use a different causal-conv1d tag or branch.
  CAUSAL_CONV1D_REF=v1.6.2.post1 INSTALL_QWEN35_FASTPATH=1 scripts/install_env.sh

Environment variables:
  ENV_NAME                  Conda env name. Default: ${ENV_NAME}
  INSTALL_QWEN35_FASTPATH   Install Qwen3.5 fast-path deps. Default: ${INSTALL_QWEN35_FASTPATH}
  QWEN35_FASTPATH_ONLY      Only add the Qwen3.5 fast path to the existing env ENV_NAME
                            (no other package is installed or upgraded). Default: ${QWEN35_FASTPATH_ONLY}
  FLA_VERSION               fla-core / flash-linear-attention version. Default: ${FLA_VERSION}
  CAUSAL_CONV1D_REF         causal-conv1d git tag or branch. Default: ${CAUSAL_CONV1D_REF}
  CUDA_TOOLKIT_VERSION      Conda CUDA toolkit version. Default: ${CUDA_TOOLKIT_VERSION}
  CUDA_LABEL                NVIDIA conda CUDA label. Default: ${CUDA_LABEL}
  INSTALL_CONDA_CUDA        Install CUDA toolkit into conda env. Default: ${INSTALL_CONDA_CUDA}
  INSTALL_DEV_TOOLS         Install pytest, ruff and pre-commit (tests and checks). Default: ${INSTALL_DEV_TOOLS}
  VLLM_VERSION              vLLM version. Default: ${VLLM_VERSION}
  FLASH_ATTN_VERSION        flash-attn version. Default: ${FLASH_ATTN_VERSION}
  CPP_RUNTIME_CHANNEL       Channel for libstdcxx-ng/libgcc-ng. Default: ${CPP_RUNTIME_CHANNEL}
  MAX_JOBS                  Parallel build jobs. Auto-detected if unset.

Notes:
  - Do not install flash-linear-attention with -U and dependency resolution in
    this env; it can upgrade torch, transformers, or triton.
  - causal-conv1d is source-built to avoid incompatible manylinux wheels on
    hosts with older glibc versions.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

if [[ "$#" -gt 0 ]]; then
    echo "[ERROR] Unknown argument: $1"
    echo ""
    usage
    exit 2
fi

compute_default_max_jobs() {
    local cpu_count
    local mem_kb
    local mem_gb
    local jobs_by_cpu
    local jobs_by_mem
    local jobs

    cpu_count="$(nproc)"
    mem_kb="$(awk '/MemTotal:/ {print $2}' /proc/meminfo 2>/dev/null || echo 0)"
    mem_gb=$((mem_kb / 1024 / 1024))

    jobs_by_cpu=$((cpu_count / 4))
    if (( jobs_by_cpu < 1 )); then
        jobs_by_cpu=1
    fi

    jobs_by_mem=$((mem_gb / 8))
    if (( jobs_by_mem < 1 )); then
        jobs_by_mem=1
    fi

    jobs=$((jobs_by_cpu < jobs_by_mem ? jobs_by_cpu : jobs_by_mem))
    if (( jobs > 8 )); then
        jobs=8
    fi

    echo "${jobs}"
}

export MAX_JOBS="${MAX_JOBS:-$(compute_default_max_jobs)}"

echo "=== EasyR1 Installation Script ==="
echo "This script creates a conda environment and installs the dependencies required by the current repository."
echo ""
echo "Repository root: ${REPO_ROOT}"
echo "Conda env name: ${ENV_NAME}"
echo "Requested Python version: ${PYTHON_VERSION}"
echo "Install CUDA toolkit into conda env: ${INSTALL_CONDA_CUDA}"
echo "C++ runtime channel: ${CPP_RUNTIME_CHANNEL}"
echo "Install Qwen3.5 fast-path dependencies: ${INSTALL_QWEN35_FASTPATH}"
echo "Only add the Qwen3.5 fast path to an existing env: ${QWEN35_FASTPATH_ONLY}"
echo "FLA version for Qwen3.5 fast path: ${FLA_VERSION}"
echo "causal-conv1d source ref: ${CAUSAL_CONV1D_REF}"
echo "MAX_JOBS for source builds: ${MAX_JOBS}"
echo ""

require_command() {
    if ! command -v "$1" >/dev/null 2>&1; then
        echo "[ERROR] Required command not found: $1"
        exit 1
    fi
}

run_conda_relaxed() {
    set +u
    conda "$@"
    set -u
}

activate_conda_env() {
    set +u
    # shellcheck disable=SC1091
    source "${CONDA_BASE}/etc/profile.d/conda.sh"
    conda activate "$1"
    set -u
}

build_filtered_requirements() {
    local filtered_requirements="${TEMP_DIR}/requirements.runtime.txt"
    grep -Ev '^(flash-attn|vllm)([<>=!~].*)?$' requirements.txt > "${filtered_requirements}"
    echo "${filtered_requirements}"
}

prepend_conda_runtime_libs() {
    if [[ -z "${CONDA_PREFIX:-}" ]]; then
        return 0
    fi

    local preload_libs=()
    if [[ -f "${CONDA_PREFIX}/lib/libstdc++.so.6" ]]; then
        preload_libs+=("${CONDA_PREFIX}/lib/libstdc++.so.6")
    fi
    if [[ -f "${CONDA_PREFIX}/lib/libgcc_s.so.1" ]]; then
        preload_libs+=("${CONDA_PREFIX}/lib/libgcc_s.so.1")
    fi

    if [[ "${#preload_libs[@]}" -eq 0 ]]; then
        return 0
    fi

    export LD_PRELOAD="$(IFS=:; echo "${preload_libs[*]}")${LD_PRELOAD:+:${LD_PRELOAD}}"
}

install_conda_runtime_hooks() {
    if [[ "${RUNTIME_HOOKS_INSTALLED}" == "1" ]]; then
        return 0
    fi

    local activate_dir="${CONDA_PREFIX}/etc/conda/activate.d"
    local deactivate_dir="${CONDA_PREFIX}/etc/conda/deactivate.d"

    mkdir -p "${activate_dir}" "${deactivate_dir}"

    cat > "${activate_dir}/easyr1-runtime-libs.sh" <<'EOF'
#!/usr/bin/env bash

export _EASYR1_OLD_LD_PRELOAD="${LD_PRELOAD-}"
unset _EASYR1_OLD_LD_LIBRARY_PATH
_easyr1_preload_libs=()
if [[ -f "${CONDA_PREFIX}/lib/libstdc++.so.6" ]]; then
    _easyr1_preload_libs+=("${CONDA_PREFIX}/lib/libstdc++.so.6")
fi
if [[ -f "${CONDA_PREFIX}/lib/libgcc_s.so.1" ]]; then
    _easyr1_preload_libs+=("${CONDA_PREFIX}/lib/libgcc_s.so.1")
fi

if [[ "${#_easyr1_preload_libs[@]}" -gt 0 ]]; then
    export LD_PRELOAD="$(IFS=:; echo "${_easyr1_preload_libs[*]}")${LD_PRELOAD:+:${LD_PRELOAD}}"
fi
EOF

    cat > "${deactivate_dir}/easyr1-runtime-libs.sh" <<'EOF'
#!/usr/bin/env bash

if [[ -n "${_EASYR1_OLD_LD_PRELOAD+x}" ]]; then
    if [[ -n "${_EASYR1_OLD_LD_PRELOAD}" ]]; then
        export LD_PRELOAD="${_EASYR1_OLD_LD_PRELOAD}"
    else
        unset LD_PRELOAD
    fi
    unset _EASYR1_OLD_LD_PRELOAD
fi

if [[ -n "${_EASYR1_OLD_LD_LIBRARY_PATH+x}" ]]; then
    if [[ -n "${_EASYR1_OLD_LD_LIBRARY_PATH}" ]]; then
        export LD_LIBRARY_PATH="${_EASYR1_OLD_LD_LIBRARY_PATH}"
    else
        unset LD_LIBRARY_PATH
    fi
    unset _EASYR1_OLD_LD_LIBRARY_PATH
fi
EOF

    chmod +x "${activate_dir}/easyr1-runtime-libs.sh" "${deactivate_dir}/easyr1-runtime-libs.sh"
    RUNTIME_HOOKS_INSTALLED=1
    NEEDS_REACTIVATE_NOTICE=1
    echo "[OK] Installed conda activate/deactivate hooks for runtime libraries"
}

ensure_conda_cpp_runtime() {
    if [[ "${CPP_RUNTIME_REFRESHED}" == "1" ]]; then
        prepend_conda_runtime_libs
        return 0
    fi

    echo "Refreshing C++ runtime inside the active conda environment..."
    run_conda_relaxed install -y -c "${CPP_RUNTIME_CHANNEL}" libstdcxx-ng libgcc-ng
    prepend_conda_runtime_libs
    CPP_RUNTIME_REFRESHED=1

    if [[ -f "${CONDA_PREFIX}/lib/libstdc++.so.6" ]]; then
        echo "[OK] Using conda libstdc++: ${CONDA_PREFIX}/lib/libstdc++.so.6"
        if command -v strings >/dev/null 2>&1 && strings "${CONDA_PREFIX}/lib/libstdc++.so.6" | grep -Fq 'CXXABI_1.3.15'; then
            echo "[OK] Conda C++ runtime provides CXXABI_1.3.15"
        else
            echo "[WARN] Could not confirm CXXABI_1.3.15 in the active conda runtime."
            echo "       flash-attn may still fail if the solver selected an older libstdc++ build."
        fi
    fi
}

verify_flash_attn_import() {
    local import_log="${1:-${TEMP_DIR}/flash_attn_import.log}"

    if python - <<'EOF' >"${import_log}" 2>&1
import flash_attn
print('flash_attn imported OK:', flash_attn.__version__)
EOF
    then
        cat "${import_log}"
        return 0
    fi

    cat "${import_log}"
    return 1
}

import_needs_cpp_runtime_fix() {
    local import_log="$1"
    grep -Eq 'CXXABI_|GLIBCXX_|libstdc\+\+\.so\.6|libgcc_s\.so\.1|version `[^`]+` not found' "${import_log}"
}

verify_flash_attn_with_runtime_fix() {
    local import_log="${TEMP_DIR}/flash_attn_import.log"

    if verify_flash_attn_import "${import_log}"; then
        return 0
    fi

    if import_needs_cpp_runtime_fix "${import_log}"; then
        echo "[WARN] flash-attn import failed with a C++ runtime error. Refreshing conda runtime and retrying..."
        ensure_conda_cpp_runtime
        if verify_flash_attn_import "${import_log}"; then
            install_conda_runtime_hooks
            return 0
        fi
        return 1
    fi

    return 1
}

install_flash_attn() {
    local success=false
    local abi_flag
    local wheel_url

    echo "Preparing flash-attn installation..."
    pip uninstall -y flash-attn 2>/dev/null || true

    if [[ "${PYTORCH_VERSION}" == 2.8.* ]] && [[ "$(uname -s)" == "Linux" ]] && [[ "$(uname -m)" == "x86_64" ]]; then
        abi_flag="$(python -c "import torch; print('TRUE' if torch._C._GLIBCXX_USE_CXX11_ABI else 'FALSE')")"
        wheel_url="https://github.com/Dao-AILab/flash-attention/releases/download/v${FLASH_ATTN_VERSION}/flash_attn-${FLASH_ATTN_VERSION}+cu12torch2.8cxx11abi${abi_flag}-${PYTHON_WHEEL_TAG}-linux_x86_64.whl"

        echo "Attempting official flash-attn wheel..."
        if pip install --no-cache-dir "${wheel_url}"; then
            if verify_flash_attn_with_runtime_fix; then
                echo "[OK] Installed flash-attn ${FLASH_ATTN_VERSION} from the official wheel"
                success=true
            else
                echo "[WARN] flash-attn wheel installed but failed to import, falling back to source build."
                pip uninstall -y flash-attn 2>/dev/null || true
            fi
        else
            echo "[WARN] Official flash-attn wheel install failed, falling back to source build."
        fi
    fi

    if [[ "${success}" == false ]]; then
        if ! command -v nvcc >/dev/null 2>&1; then
            echo "[ERROR] nvcc is not available, and no compatible flash-attn wheel could be installed."
            echo "        Set INSTALL_CONDA_CUDA=1 or provide a system CUDA toolkit before rerunning this script."
            return 1
        fi

        echo "Attempting flash-attn source build from PyPI..."
        if uv pip install --no-build-isolation --no-cache-dir "flash-attn==${FLASH_ATTN_VERSION}"; then
            if verify_flash_attn_with_runtime_fix; then
                echo "[OK] Installed flash-attn ${FLASH_ATTN_VERSION} from source"
                success=true
            else
                echo "[WARN] PyPI source build completed but flash-attn still failed to import."
                pip uninstall -y flash-attn 2>/dev/null || true
            fi
        fi
    fi

    if [[ "${success}" == false ]]; then
        echo "Attempting flash-attn source build from the upstream repository..."
        git clone -b "v${FLASH_ATTN_VERSION}" --single-branch https://github.com/Dao-AILab/flash-attention.git "${TEMP_DIR}/flash-attention"
        (
            cd "${TEMP_DIR}/flash-attention"
            python setup.py install
        )
        if verify_flash_attn_with_runtime_fix; then
            echo "[OK] Installed flash-attn ${FLASH_ATTN_VERSION} from the upstream repository"
            success=true
        else
            echo "[ERROR] Upstream flash-attn source build completed but the module still failed to import."
            return 1
        fi
    fi

    if [[ "${success}" == false ]]; then
        echo "[ERROR] Failed to install a working flash-attn build."
        return 1
    fi
}

install_qwen35_fastpath() {
    if ! command -v nvcc >/dev/null 2>&1; then
        echo "[ERROR] nvcc is required to build causal-conv1d from source."
        echo "        Set INSTALL_CONDA_CUDA=1 or provide a system CUDA toolkit before rerunning this script."
        return 1
    fi

    echo "Installing Qwen3.5 flash-linear-attention packages without dependency resolution..."
    uv pip install --no-deps "fla-core==${FLA_VERSION}" "flash-linear-attention==${FLA_VERSION}"

    echo "Building causal-conv1d from source (${CAUSAL_CONV1D_REF})..."
    pip uninstall -y causal-conv1d causal_conv1d 2>/dev/null || true
    git clone --depth 1 --branch "${CAUSAL_CONV1D_REF}" \
        https://github.com/Dao-AILab/causal-conv1d.git \
        "${TEMP_DIR}/causal-conv1d"

    (
        cd "${TEMP_DIR}/causal-conv1d"
        CAUSAL_CONV1D_FORCE_BUILD=TRUE \
        MAX_JOBS="${MAX_JOBS}" \
        UV_NO_CACHE=1 \
        uv pip install -v --no-build-isolation --no-deps .
    )
}

# Imports the fast-path kernels as transformers does; succeeds only when Qwen3.5 can use them.
verify_qwen35_fastpath_import() {
    local import_log="${1:-${TEMP_DIR}/qwen35_fastpath_import.log}"

    if python - <<'EOF' >"${import_log}" 2>&1
import causal_conv1d
import fla
from transformers.models.qwen3_next import modeling_qwen3_next as qwen3_next

if not qwen3_next.is_fast_path_available:
    raise SystemExit("transformers does not see the Qwen3.5 fast path")
print("Qwen3.5 fast path imported OK: fla", fla.__version__)
EOF
    then
        cat "${import_log}"
        return 0
    fi

    cat "${import_log}"
    return 1
}

# causal-conv1d is built with the conda compiler, so like flash-attn it can need the conda C++ runtime
# (and the activation hook that loads it) even when flash-attn itself did not.
verify_qwen35_fastpath_with_runtime_fix() {
    local import_log="${TEMP_DIR}/qwen35_fastpath_import.log"

    if verify_qwen35_fastpath_import "${import_log}"; then
        return 0
    fi

    if import_needs_cpp_runtime_fix "${import_log}"; then
        echo "[WARN] The Qwen3.5 fast path failed to import with a C++ runtime error. Refreshing conda runtime and retrying..."
        ensure_conda_cpp_runtime
        if verify_qwen35_fastpath_import "${import_log}"; then
            install_conda_runtime_hooks
            return 0
        fi
    fi

    return 1
}

pinned_package_versions() {
    python -c "import importlib.metadata as m; print(' '.join(f'{p}=={m.version(p)}' for p in ('torch', 'transformers', 'triton')))"
}

# QWEN35_FASTPATH_ONLY=1: add the fast path to an environment installed earlier, changing nothing else.
add_qwen35_fastpath_to_existing_env() {
    echo "=== Adding the Qwen3.5 fast path to the existing conda environment ${ENV_NAME} ==="
    if ! conda env list | awk 'NF > 0 && $1 !~ /^#/' | awk '{print $1}' | grep -Fxq "${ENV_NAME}"; then
        echo "[ERROR] Conda environment ${ENV_NAME} does not exist. Create it with the fast path instead:"
        echo "        ENV_NAME=${ENV_NAME} INSTALL_QWEN35_FASTPATH=1 bash scripts/install_env.sh"
        exit 1
    fi

    CONDA_BASE="$(conda info --base)"
    activate_conda_env "${ENV_NAME}"
    echo "[OK] Active conda environment: ${CONDA_DEFAULT_ENV}"

    local versions_before versions_after pinned_transformers
    versions_before="$(pinned_package_versions)"
    echo "[OK] Installed: ${versions_before}"
    pinned_transformers="$(grep -E '^transformers==' "${REPO_ROOT}/scripts/constraints.txt" || true)"
    if [[ -n "${pinned_transformers}" && " ${versions_before} " != *" ${pinned_transformers} "* ]]; then
        echo "[WARN] This repository is tested with ${pinned_transformers}; fla ${FLA_VERSION} may not match other versions."
    fi

    if verify_qwen35_fastpath_with_runtime_fix; then
        echo "[OK] The Qwen3.5 fast path already works in ${ENV_NAME}; nothing to install."
    else
        echo "Installing the build tools the source build needs (installed versions are kept)..."
        python -m pip install uv ninja packaging
        if ! command -v nvcc >/dev/null 2>&1; then
            if [[ "${INSTALL_CONDA_CUDA}" == "1" ]]; then
                echo "nvcc not found: installing CUDA toolkit ${CUDA_TOOLKIT_VERSION} into ${ENV_NAME}..."
                run_conda_relaxed install -y -c "nvidia/label/${CUDA_LABEL}" "cuda-toolkit=${CUDA_TOOLKIT_VERSION}"
            fi
        fi
        install_qwen35_fastpath
        if ! verify_qwen35_fastpath_with_runtime_fix; then
            echo "[ERROR] The Qwen3.5 fast path was installed but cannot be imported (see the log above)."
            exit 1
        fi
        versions_after="$(pinned_package_versions)"
        if [[ "${versions_after}" != "${versions_before}" ]]; then
            echo "[ERROR] Installing the fast path changed core packages:"
            echo "        before: ${versions_before}"
            echo "        after:  ${versions_after}"
            exit 1
        fi
    fi

    echo ""
    echo "Verifying installation..."
    export INSTALL_QWEN35_FASTPATH
    verify_installation

    echo ""
    echo "[OK] The Qwen3.5 fast path works in conda env: ${ENV_NAME}"
    if [[ "${NEEDS_REACTIVATE_NOTICE}" == "1" ]]; then
        echo ""
        echo "[NOTE] This run installed runtime-library hooks for ${ENV_NAME}; shells that already had it"
        echo "       active need: conda deactivate && conda activate ${ENV_NAME}"
    fi
}

verify_installation() {
    local verify_script="${TEMP_DIR}/verify_easyr1.py"

    cat > "${verify_script}" <<'EOF'
import importlib
import os
import traceback

modules = [
    ("torch", False),
    ("transformers", False),
    ("vllm", False),
    ("flash_attn", False),
    ("liger_kernel", False),
    ("qwen_vl_utils", False),
    ("verl", False),
]

all_ok = True

for module_name, optional in modules:
    try:
        module = importlib.import_module(module_name)
        version = getattr(module, "__version__", "unknown")
        print(f"[OK] {module_name} import succeeded (version: {version})")
    except Exception as exc:
        if optional:
            print(f"[WARN] {module_name} import failed: {exc}")
        else:
            all_ok = False
            print(f"[ERROR] {module_name} import failed: {exc}")
            traceback.print_exc()

import torch
print(f"[INFO] torch.cuda.is_available(): {torch.cuda.is_available()}")
print(f"[INFO] torch version: {torch.__version__}")
print(f"[INFO] torch CUDA runtime: {torch.version.cuda}")

if os.environ.get("INSTALL_QWEN35_FASTPATH", "0") == "1":
    try:
        from transformers.utils.import_utils import (
            is_causal_conv1d_available,
            is_flash_linear_attention_available,
        )

        fla_available = is_flash_linear_attention_available()
        causal_conv1d_available = is_causal_conv1d_available()
        print(f"[INFO] Qwen3.5 flash-linear-attention available: {fla_available}")
        print(f"[INFO] Qwen3.5 causal-conv1d available: {causal_conv1d_available}")
        if not fla_available or not causal_conv1d_available:
            all_ok = False
            print("[ERROR] Qwen3.5 fast-path dependency availability check failed.")

        from transformers.models.qwen3_next import modeling_qwen3_next as qwen3_next

        qwen35_checks = {
            "is_fast_path_available": bool(qwen3_next.is_fast_path_available),
            "causal_conv1d_fn": qwen3_next.causal_conv1d_fn is not None,
            "chunk_gated_delta_rule": qwen3_next.chunk_gated_delta_rule is not None,
            "fused_recurrent_gated_delta_rule": qwen3_next.fused_recurrent_gated_delta_rule is not None,
        }
        for name, ok in qwen35_checks.items():
            print(f"[INFO] Qwen3.5 {name}: {ok}")
        if not all(qwen35_checks.values()):
            all_ok = False
            print("[ERROR] Qwen3.5 fast-path kernel import check failed.")
    except Exception as exc:
        all_ok = False
        print(f"[ERROR] Qwen3.5 fast-path verification failed: {exc}")
        traceback.print_exc()

if not all_ok:
    raise SystemExit(1)
EOF

    python "${verify_script}"
}

require_command conda
require_command git
require_command grep
require_command awk

if [[ "$(uname -s)" != "Linux" ]]; then
    echo "[ERROR] This script currently targets Linux environments."
    exit 1
fi

if [[ "${QWEN35_FASTPATH_ONLY}" == "1" ]]; then
    add_qwen35_fastpath_to_existing_env
    exit 0
fi

echo "=== Environment Detection ==="

echo "Step 1/8: Creating or reusing conda environment..."
if conda env list | awk 'NF > 0 && $1 !~ /^#/' | awk '{print $1}' | grep -Fxq "${ENV_NAME}"; then
    echo "[OK] Reusing existing conda environment: ${ENV_NAME}"
else
    run_conda_relaxed create -n "${ENV_NAME}" "python=${PYTHON_VERSION}" -y
fi

CONDA_BASE="$(conda info --base)"
activate_conda_env "${ENV_NAME}"

CURRENT_PYTHON_VERSION="$(python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')")"
CURRENT_PYTHON_MM="$(python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")"
EXPECTED_PYTHON_MM="${PYTHON_VERSION}"

if [[ "${CURRENT_PYTHON_MM}" != "${EXPECTED_PYTHON_MM}" ]]; then
    echo "[ERROR] Existing environment ${ENV_NAME} uses Python ${CURRENT_PYTHON_MM}, but ${EXPECTED_PYTHON_MM} was requested."
    echo "        Remove the environment or rerun with a different ENV_NAME / PYTHON_VERSION."
    exit 1
fi

echo "[OK] Active conda environment: ${CONDA_DEFAULT_ENV}"
echo "[OK] Active Python version: ${CURRENT_PYTHON_VERSION}"

echo ""
echo "Step 2/8: Installing or refreshing basic build tools..."
python -m pip install --upgrade pip setuptools wheel packaging ninja uv
uv --version

echo ""
echo "Step 3/8: Installing CUDA toolkit 12.8 inside the conda environment..."
if [[ "${INSTALL_CONDA_CUDA}" == "1" ]]; then
    run_conda_relaxed install -y -c "nvidia/label/${CUDA_LABEL}" "cuda-toolkit=${CUDA_TOOLKIT_VERSION}"
    which nvcc
    nvcc -V
else
    echo "[WARN] Skipping conda CUDA toolkit install because INSTALL_CONDA_CUDA=${INSTALL_CONDA_CUDA}"
fi

echo ""
echo "Step 4/8: Installing vLLM and its CUDA 12.8 torch backend..."
uv pip install --upgrade "vllm==${VLLM_VERSION}" --torch-backend=cu128

PYTORCH_VERSION="$(python -c "import torch; print(torch.__version__.split('+')[0])")"
TORCH_CUDA_VERSION="$(python -c "import torch; print(torch.version.cuda)")"
echo "[OK] PyTorch version: ${PYTORCH_VERSION}"
echo "[OK] PyTorch CUDA runtime: ${TORCH_CUDA_VERSION}"

echo ""
echo "Step 5/8: Installing repository runtime dependencies..."
FILTERED_REQUIREMENTS="$(build_filtered_requirements)"
uv pip install -r "${FILTERED_REQUIREMENTS}" -c "${REPO_ROOT}/scripts/constraints.txt"

echo ""
echo "Step 6/8: Installing flash-attn and applying C++ runtime fixes only if needed..."
install_flash_attn
uv pip install --upgrade "nvidia-cudnn-cu12>=9.15"

echo ""
echo "Step 7/8: Installing Qwen3.5 fast-path dependencies..."
if [[ "${INSTALL_QWEN35_FASTPATH}" == "1" ]]; then
    install_qwen35_fastpath
    if ! verify_qwen35_fastpath_with_runtime_fix; then
        echo "[ERROR] The Qwen3.5 fast path was installed but cannot be imported (see the log above)."
        exit 1
    fi
else
    echo "[WARN] Skipping Qwen3.5 fast-path dependencies because INSTALL_QWEN35_FASTPATH=${INSTALL_QWEN35_FASTPATH}"
fi

echo ""
echo "Step 8/8: Installing EasyR1 in editable mode..."
uv pip install --no-deps -e .
if [[ "${INSTALL_DEV_TOOLS}" == "1" ]]; then
    # unit tests (python -m pytest -q tests/) and the checks (pre-commit run --all-files, make quality)
    uv pip install pytest ruff pre-commit -c "${REPO_ROOT}/scripts/constraints.txt"
else
    echo "[WARN] Skipping pytest/ruff/pre-commit because INSTALL_DEV_TOOLS=${INSTALL_DEV_TOOLS}"
fi

echo ""
echo "Verifying installation..."
export INSTALL_QWEN35_FASTPATH
verify_installation

echo ""
echo "Installation summary:"
echo "[OK] EasyR1 environment is ready in conda env: ${ENV_NAME}"
if [[ "${INSTALL_QWEN35_FASTPATH}" == "1" ]]; then
    echo "[OK] Verified torch / transformers / vllm / flash-attn / Qwen3.5 fast path / verl imports"
else
    echo "[OK] Verified torch / transformers / vllm / flash-attn / verl imports"
fi
echo ""
echo "To use this environment later, run:"
echo "  conda activate ${ENV_NAME}"
if [[ "${NEEDS_REACTIVATE_NOTICE}" == "1" ]]; then
    echo ""
    echo "[NOTE] This run installed runtime-library hooks for ${ENV_NAME}."
    echo "       Your first normal 'conda activate ${ENV_NAME}' will load them automatically."
    echo "       Only shells that had ${ENV_NAME} active before this installation need reactivation:"
    echo "         conda deactivate && conda activate ${ENV_NAME}"
fi
