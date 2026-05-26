#!/usr/bin/env bash
set -euo pipefail

CHECKPOINT=${CHECKPOINT:-/home/liujinxiu/xAR_copy/xAR-B.pth}
VAE_PATH=${VAE_PATH:-/home/liujinxiu/xAR_copy/vae/kl16.ckpt}
ARCH=${ARCH:-base}
DEVICE=${DEVICE:-cuda}
OUT_DIR=${OUT_DIR:-generated}

mkdir -p "${OUT_DIR}"

sum=0
n=0

for i in $(seq 1 10); do
  out=$(python generate_single_image_patchschedule.py \
    --checkpoint "${CHECKPOINT}" \
    --vae-path "${VAE_PATH}" \
    --arch "${ARCH}" \
    --device "${DEVICE}" \
    --output "${OUT_DIR}/patchschedule_${i}.png")
  echo "${out}"
  t=$(echo "${out}" | rg -o "Sample time \\(model-only\\): ([0-9.]+)s" -r '$1' || true)
  if [ -n "${t}" ]; then
    sum=$(python - <<PY
s=${sum}
t=${t}
print(s+t)
PY
)
    n=$((n+1))
  fi
done

python - <<PY
s=${sum}
n=${n}
print(f"Average sample time (model-only) over {n} runs: {s/n if n else 0:.2f}s")
PY
