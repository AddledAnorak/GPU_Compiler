#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD="${MGC_BUILD_DIR:-$ROOT/mesa/build}"
export PATH="$ROOT/.venv/bin:$PATH"

if [ ! -f "$ROOT/mesa/src/gallium/drivers/mgc/meson.build" ]; then
    cat >&2 <<'EOF'
A clean Mesa 25.2.0 checkout is required at mesa/. From inside mesa/:
  git am ../mesa-patches/0001-*.patch ../mesa-patches/0002-*.patch
  git apply ../mesa-patches/0003-*.patch ../mesa-patches/0004-*.patch
EOF
    exit 1
fi
if [ ! -f "$BUILD/build.ninja" ]; then
    meson setup "$BUILD" "$ROOT/mesa" \
        -Dgallium-drivers=softpipe,mgc -Dvulkan-drivers= -Dplatforms= \
        -Degl-native-platform=surfaceless -Dglx=disabled -Dgbm=disabled \
        -Dllvm=disabled -Dbuildtype=debugoptimized
fi
ninja -C "$BUILD"
export PKG_CONFIG_PATH="$BUILD/meson-uninstalled${PKG_CONFIG_PATH:+:$PKG_CONFIG_PATH}"
read -r -a EGL_FLAGS <<< "$(pkg-config --cflags --libs egl glesv2)"
for HARNESS in mgc_harness nir_dump_harness render_phong_demo; do
    "${CC:-cc}" "$ROOT/tools/$HARNESS.c" -o "$ROOT/tools/$HARNESS" "${EGL_FLAGS[@]}" -lm
done
printf 'Mesa and three GLES harnesses built successfully in %s\n' "$BUILD"
