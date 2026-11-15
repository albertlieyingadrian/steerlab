#!/usr/bin/env bash
# Rebuild every number in the looped-transformer note. Training and probes run on Modal (L4, at most 3 containers,
# about $13 in all); analysis, figures, the demo data and the cover run on the laptop in seconds.
#   bash experiments/looped-transformer/reproduce.sh            # run everything
#   bash experiments/looped-transformer/reproduce.sh --dry-run  # print the commands only
# Set MODAL_PROFILE to your Modal profile. Modal client 1.6.x needs MODAL_IMAGE_BUILDER_VERSION=2025.06 on
# workspaces whose default image builder is newer.
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=.venv/bin/python
MODAL=.venv/bin/modal
D=experiments/looped-transformer
DRY=0
[[ "${1:-}" == "--dry-run" ]] && DRY=1
run() { echo "+ $*"; if [[ "$DRY" == 0 ]]; then "$@"; fi; }

# Pre-registered runs: 5 models x 3 seeds, 20,000 steps each (curriculum to 8 hops)
run $MODAL run $D/modal_train.py --models looped,standard4,standard10,looped_fixed4,looped_noinject --seeds 1,2,3 --steps 20000
# Amendment 1 (exploratory): training hops up to 16
run $MODAL run $D/modal_train.py --models looped,standard4,standard10 --seeds 1,2,3 --steps 20000 --extra "--max-hops 16 --tag _h16"
# Upload the checkpoints once, then probes (18 models) and the twin-distractor control set (24 models, amendment 2),
# each packed into one container
run $MODAL volume put -f looped-transformer-ckpt $D/results/runs/ /
run $MODAL run $D/modal_train.py --task packed-inside --runs standard4_s1,standard4_s2,standard4_s3,looped_s1,looped_s2,looped_s3,standard10_s1,standard10_s2,standard10_s3,looped_fixed4_s1,looped_fixed4_s2,looped_fixed4_s3,looped_noinject_s1,looped_noinject_s2,looped_noinject_s3,looped_h16_s1,looped_h16_s2,looped_h16_s3
run $MODAL run $D/modal_train.py --task packed-control --runs standard4_s1,standard4_s2,standard4_s3,looped_s1,looped_s2,looped_s3,standard10_s1,standard10_s2,standard10_s3,looped_fixed4_s1,looped_fixed4_s2,looped_fixed4_s3,looped_noinject_s1,looped_noinject_s2,looped_noinject_s3,standard4_h16_s1,standard4_h16_s2,standard4_h16_s3,looped_h16_s1,looped_h16_s2,looped_h16_s3,standard10_h16_s1,standard10_h16_s2,standard10_h16_s3

# Numbers, figures, demo data, cover (laptop, seconds)
run $PY $D/analyse.py
run $PY $D/make_figures.py
run $PY $D/make_demo.py
run $PY $D/make_cover.py
