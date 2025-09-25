#!/usr/bin/env bash
set -euo pipefail

# 该脚本用于下载 MAESTRO v3 与 Slakh2100 数据集，并根据 config 中的路径组织目录结构。
# 由于实际下载需要大量网络/存储资源，这里仅给出流程模板。

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <output_root>" >&2
  exit 1
fi

OUTPUT_ROOT="$1"
mkdir -p "${OUTPUT_ROOT}"

cat <<SCRIPT_NOTE
[prepare_datasets] 请按照以下步骤手动完成：
1. 下载 MAESTRO v3.0.0 官方压缩包，将音频与 MIDI 解压至 \\"${OUTPUT_ROOT}/maestro_v3\\"。
2. 下载 Slakh2100 数据集，放置在 \\"${OUTPUT_ROOT}/slakh2100\\"。
3. 运行 src/hs_mad/data/datasets/build_manifests.py 生成 manifest JSON。
SCRIPT_NOTE
