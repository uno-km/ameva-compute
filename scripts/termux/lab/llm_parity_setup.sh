#!/data/data/com.termux/files/usr/bin/bash
# Build ameva_llm_parity / ameva_quantize (into OUT, default ~/abi-lab/out-g2) against the llama.h of the installed
# libllama (b8471) and fetch the corpus.
set -u
LAB=$HOME/abi-lab
SRC=$LAB/src
P=/data/data/com.termux/files/usr
OUT=${OUT:-$LAB/out-g2}
GLIB="$P/lib"
[ -d "$LAB/rt098" ] && GLIB="$LAB/rt098"
mkdir -p "$LAB/llama-b8471/include" "$LAB/corpus" "$OUT"

# the installed libllama must be build 8471: the header below is that build's
strings "$P/lib/libllama.so.0" | grep -q "llama_model_quantize" || { echo "LIBLLAMA_MISSING"; exit 1; }
cmp -s "$P/lib/libllama.so.0" "$P/lib/libllama.so.0.0.8471" || { echo "LIBLLAMA_NOT_8471"; exit 1; }
H=$LAB/llama-b8471/include/llama.h
if [ ! -s "$H" ]; then
    curl -sSL -o "$H" https://raw.githubusercontent.com/ggml-org/llama.cpp/b8471/include/llama.h || { echo HEADER_FETCH_FAIL; exit 1; }
fi
echo "llama.h b8471: $(wc -c < "$H") bytes"

for t in tests/ameva_llm_parity tools/ameva_quantize; do
    clang++ -O2 -std=c++17 -Wall -Wextra -I "$LAB/llama-b8471/include" -I "$LAB/ggml-b8471/include" "$SRC/$t.cpp" \
        "$P/lib/libllama.so.0" "$GLIB/libggml-base.so.0" "$GLIB/libggml.so.0" -ldl -Wl,-rpath,"$P/lib" \
        -o "$OUT/$(basename "$t")" || { echo "BUILD_FAIL $t"; exit 1; }
done
readelf -d "$OUT/ameva_llm_parity" | grep -E "NEEDED|RUNPATH"

C=$LAB/corpus
if [ ! -s "$C/wiki.test.raw" ]; then
    curl -sSL -o "$C/wikitext-2-raw-v1.zip" https://huggingface.co/datasets/ggml-org/ci/resolve/main/wikitext-2-raw-v1.zip \
        || { echo CORPUS_FETCH_FAIL; exit 1; }
    unzip -o -j "$C/wikitext-2-raw-v1.zip" '*/wiki.test.raw' -d "$C" > /dev/null || { echo CORPUS_UNZIP_FAIL; exit 1; }
fi
echo "corpus: $(wc -c < "$C/wiki.test.raw") bytes, sha256 $(sha256sum "$C/wiki.test.raw" | cut -c1-16)"
echo SETUP_OK
