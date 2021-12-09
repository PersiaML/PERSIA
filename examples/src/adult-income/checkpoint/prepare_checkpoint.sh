#!/bin/bash
set -x

export PYTHONDONTWRITEBYTECODE=1

SCRIPTPATH="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
export PERSIA_CKPT_DIR=${SCRIPTPATH}/adult_income_ckpt/

honcho start -e .honcho.env