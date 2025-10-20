#!/bin/bash -e

print_usage () {
    echo "Usage: $0 [--revert]"
}

MODE="apply"

if [ $# -gt 1 ]; then
    print_usage
    exit 1
fi

if [ $# -eq 1 ]; then
    if [ "$1" == "--revert" ]; then
	MODE="revert"
    else
	print_usage
	exit 1
    fi
fi

PIP_SHOW_RESULT=$(pip show vllm)
VLLM_DIR=$(echo "$PIP_SHOW_RESULT" | awk '/Location:/{print $2}')/vllm
VLLM_VER=$(echo "$PIP_SHOW_RESULT" | awk '/Version:/{print $2}')
VLLM_VER=${VLLM_VER%%+*} # remove string after '+' (e.g. "0.6.4.post1+cu124" -> "0.6.4.post1")

SCRIPT_DIR=$(dirname ${BASH_SOURCE})
PATCH_FILE="${SCRIPT_DIR}/v${VLLM_VER}.patch"

if [ ! -f ${PATCH_FILE} ]; then
    echo "Version ${VLLM_VER} is not supported"
    exit 1
fi

if [ ${MODE} == "apply" ]; then
    patch -N -p 1 -d ${VLLM_DIR} < ${PATCH_FILE}
    cp ${SCRIPT_DIR}/aga.py ${VLLM_DIR}/
    echo "Patch applied successfully."
else
    patch -N -p 1 -d ${VLLM_DIR} -R < ${PATCH_FILE}
    rm -f ${VLLM_DIR}/aga.py
    echo "Patch reverted successfully."
fi
