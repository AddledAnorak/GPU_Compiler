#!/usr/bin/env bash
# Captures GLSL -> NIR traces for every shader pair in shaders/, using
# Mesa's softpipe reference driver (see Step 1-2 of the project plan).
#
# Requires a local Mesa build (see Phase 0 of the plan):
#   cd mesa && meson setup build -Dgallium-drivers=softpipe \
#     -Dvulkan-drivers= -Dplatforms= -Degl-native-platform=surfaceless \
#     -Dglx=disabled -Dgbm=disabled -Dllvm=disabled -Dbuildtype=debugoptimized
#   ninja -C build
# and tools/nir_dump_harness built against it -- see tools/nir_dump_harness.c
# for the build command. Both bison and flex must be installed for the
# meson setup step to succeed.
#
# Usage: tools/dump_nir.sh [build_dir]
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${1:-$ROOT/mesa/build}"
OUT_DIR="$ROOT/docs/glsl-to-nir/traces"

mkdir -p "$OUT_DIR"

if [ ! -d "$BUILD_DIR" ]; then
    echo "Mesa build not found at $BUILD_DIR." >&2
    echo "Configure it first, e.g.:" >&2
    echo "  cd $ROOT/mesa && meson setup build -Dgallium-drivers=softpipe \\" >&2
    echo "    -Dvulkan-drivers= -Dplatforms= -Degl-native-platform=surfaceless \\" >&2
    echo "    -Dglx=disabled -Dgbm=disabled -Dllvm=disabled -Dbuildtype=debugoptimized" >&2
    exit 1
fi

if [ ! -x "$ROOT/tools/nir_dump_harness" ]; then
    echo "tools/nir_dump_harness not built. Build it against this Mesa checkout:" >&2
    echo "  PKG_CONFIG_PATH=\"$BUILD_DIR/meson-uninstalled\" gcc $ROOT/tools/nir_dump_harness.c \\" >&2
    echo "    -o $ROOT/tools/nir_dump_harness \$(pkg-config --cflags --libs egl glesv2)" >&2
    exit 1
fi

# A minimal offscreen EGL harness (tools/nir_dump_harness.c) compiles+links
# vert.glsl/frag.glsl and issues one draw to exercise softpipe. Mesa first
# calls glsl_to_nir during glCompileShader; NIR_DEBUG=print emits the initial
# translation and many later optimization/driver snapshots.
for dir in "$ROOT"/shaders/*/; do
    name="$(basename "$dir")"
    echo "=== $name ==="
    NIR_DEBUG=print \
    MESA_SHADER_CACHE_DISABLE=1 \
    GALLIUM_DRIVER=softpipe \
    EGL_PLATFORM=surfaceless \
    LIBGL_ALWAYS_SOFTWARE=1 \
    LD_LIBRARY_PATH="$BUILD_DIR/src/egl:$BUILD_DIR/src/mapi/es2api:$BUILD_DIR/src/gallium/targets/dri:${LD_LIBRARY_PATH:-}" \
        "$ROOT/tools/nir_dump_harness" "$dir/vert.glsl" "$dir/frag.glsl" \
        > "$OUT_DIR/$name.nir.txt" 2>&1 || {
            echo "  (failed -- see $OUT_DIR/$name.nir.txt)"
        }
done

echo "Traces written to $OUT_DIR/"
echo "Next: regenerate docs/glsl-to-nir/dictionary.md from these traces."
echo "Week 7-8 check: python3 tools/validate_week78_nir.py"
