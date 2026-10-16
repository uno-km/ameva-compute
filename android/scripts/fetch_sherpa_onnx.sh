#!/usr/bin/env bash
# Downloads the pinned upstream k2-fsa/sherpa-onnx AAR into android/libs/ and verifies its sha256.
# Used by CI (android-ci.yml, android-release.yml) and by developers on a fresh clone. The AAR
# (~48 MiB) is deliberately not committed; android/build.gradle.kts fails fast if it is missing.
#
# Fails (non-zero) on download error or digest mismatch; never falls back to an unverified file.
set -euo pipefail

VERSION="1.13.8"
SHA256="633c24321e06b1fe79feafa03ea16cbc0f8a286641e2da3559bac91bdb13bd96"
NAME="sherpa-onnx-${VERSION}.aar"
URL="https://github.com/k2-fsa/sherpa-onnx/releases/download/v${VERSION}/${NAME}"

DEST_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/libs"
DEST="${DEST_DIR}/${NAME}"
mkdir -p "${DEST_DIR}"

verify() { echo "${SHA256}  ${1}" | sha256sum --check --status; }

if [ -f "${DEST}" ] && verify "${DEST}"; then
  echo "AMU_OK: ${NAME} present, sha256 verified"
  exit 0
fi

TMP="$(mktemp "${DEST_DIR}/.${NAME}.XXXXXX")"
trap 'rm -f "${TMP}"' EXIT
echo "Downloading ${URL}"
curl --fail --location --silent --show-error --retry 3 --retry-delay 5 --output "${TMP}" "${URL}"

if ! verify "${TMP}"; then
  echo "AMU_ERR_SHA256_MISMATCH: ${NAME} expected ${SHA256}, got $(sha256sum "${TMP}" | cut -d' ' -f1)" >&2
  exit 1
fi

mv -f "${TMP}" "${DEST}"
trap - EXIT
echo "AMU_OK: ${NAME} downloaded, sha256 verified"
