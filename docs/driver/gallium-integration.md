# Gallium Driver Basics, and Where Zynq Fits

Student C's Week 1-2 deliverable: "Learn basics of Gallium drivers and Zynq
ARM-FPGA communication." Written from reading the Mesa 25.2.0 sources while
designing the `mgc` driver skeleton (see `docs/isa/OPEN_QUESTIONS.md` and
`docs/isa/isa.yaml` for the target ISA this backend maps onto).

## 1. The `pipe_screen` / `pipe_context` split

Every Gallium driver implements two C structs, both defined in
`mesa/src/gallium/include/pipe/`:

- **`pipe_screen`** (`p_screen.h`) — one instance per physical/logical
  device. Answers capability questions ("does this GPU support NPOT
  textures?", "what's the max texture size?") and creates resources and
  contexts. Long-lived, effectively global.
- **`pipe_context`** (`p_context.h`) — one instance per rendering context
  (roughly, per `EGLContext`). Holds bound state (shaders, blend mode,
  vertex buffers, framebuffer) and is where actual work -- `draw_vbo`,
  `clear`, `create_fs_state` -- happens. Mesa's state tracker
  (`src/mesa/state_tracker/`) translates OpenGL/GLES calls into calls on
  this struct.

The state tracker is deliberately IHV-agnostic: it never touches hardware
registers or memory directly. It calls `pipe_context` methods, and it is the
driver's job underneath those methods to actually talk to the device. This
is the seam our project lives in -- we write a `pipe_screen` +
`pipe_context` implementation (`mgc_screen.c`, `mgc_context.c`) that, instead
of programming a real GPU's command ring, will eventually push instruction
words assembled by `tools/mgcasm` across to the FPGA fabric.

## 2. Where NIR fits in this split

`pipe_screen::get_compiler_options` tells the state tracker what NIR
transformations the driver wants applied before it ever sees a shader (see
the corrected table in `docs/isa/nir_opcode_triage.md` for what `mgc` asks
for and why). The shader itself arrives as NIR in
`pipe_context::create_fs_state` / `create_vs_state`, called from
`st_program.c` once GLSL linking finishes (`glLinkProgram`). This is the one
callback in the whole driver that does compiler work; everything else on
`pipe_context` is state tracking and (eventually) command submission.

For a driver whose hardware doesn't exist yet, this split is convenient: the
compiler path (`create_fs_state` walking NIR, emitting MGC assembly text via
`tools/mgcasm`) can be built, run, and tested with `tools/mgcsim` completely
independent of whether `draw_vbo` or any other rendering hook ever does real
work. See the Weeks 3-4 plan for the concrete scope cut this enables: a
"software-winsys" driver selected by `GALLIUM_DRIVER=mgc`, reached over a
*surfaceless* EGL context (no window, no pbuffer, no real framebuffer) so
`eglCreateContext` and `glLinkProgram` succeed and call our backend, without
needing anything past that to work yet.

## 3. Where a real driver would hand off to hardware

On a real GPU driver (e.g. `freedreno` for Adreno, `panfrost` for Mali), the
handoff from `pipe_context` to hardware happens roughly here:

1. **Command batching.** As the state tracker calls `pipe_context` methods
   (`set_framebuffer_state`, `bind_fs_state`, `draw_vbo`, ...), the driver
   accumulates a command buffer in a driver-private format -- register
   writes, descriptor tables, and (for us) the assembled MGC instruction
   stream for the currently bound shaders.
2. **Buffer/resource upload.** Anything the GPU needs to read -- textures,
   vertex data, the compiled shader binary itself -- lives in a
   `pipe_resource`, backed on real hardware by GPU-visible memory (allocated
   via the kernel DRM driver, e.g. through GEM/TTM on Linux). Our software
   -winsys skeleton fakes this with a plain `malloc`'d buffer
   (`mgc_resource.c`) since there's no device memory to allocate from yet.
3. **Submission.** The command buffer + resource list gets submitted to the
   kernel driver (an `ioctl` on `/dev/dri/renderD*` for a real DRM driver),
   which schedules it on the GPU and signals a fence when it completes.
   `pipe_context::flush` is roughly this trigger point; `pipe_fence_handle`
   is the completion signal the state tracker can wait on.

None of step 3 exists for `mgc` yet -- there is no kernel driver, no DRM
device node, no PCIe/AXI transaction happening. That is explicitly Week 9+
territory once the FPGA bitstream and a minimal driver on the Zynq's ARM
cores exist. Weeks 1-4 stop at step 1: prove the compiler path produces
correct MGC assembly, verified by running it on `tools/mgcsim` instead of
real silicon.

## 4. Where Zynq's ARM-FPGA link will eventually sit

The Zynq SoC pairs a hard ARM core (the "PS", processing system) with FPGA
fabric (the "PL", programmable logic) on one die, connected by several AXI
buses. For this project, the eventual (not yet Week 1-4) picture is:

- **Mesa + our Gallium driver run on the ARM cores**, under Linux, exactly
  like any other Gallium driver runs on a CPU today -- nothing about the
  state-tracker/Gallium split changes because the "GPU" happens to live in
  the FPGA fabric instead of on a separate PCIe card.
- **The MGC-FP16-v1 core is synthesized into the PL fabric.** From the ARM
  side it looks like a memory-mapped peripheral: instruction memory, data
  memory, and the framebuffer region described (with placeholder addresses)
  in `tools/mgcsim/memory.py` all become addresses on an AXI bus the ARM
  cores can read/write directly (AXI-Lite for control/status registers,
  AXI-Full or AXI-DMA for bulk instruction/data transfer).
- **A minimal kernel driver (or UIO/`/dev/mem` userspace mapping)** is what
  turns "write these bytes to this AXI address range" into something our
  `pipe_context::flush` can call. This is the Week 9+ "driver" in the
  project-plan sense -- distinct from the Gallium driver, which is Mesa's
  vocabulary for the compiler+state-tracking layer this document is about.
  Concretely, `store_addr`/`store`/`load_addr`/`load` in the MGC ISA
  (`docs/isa/isa.yaml`, channels 17-20) are the instructions our compiled
  shaders use to talk to that same memory-mapped region from *inside* the
  running MGC program, once it's issued.
- **Program start/stop is host-polled** per the current best guess
  (`docs/isa/ENCODING_GUESSES.md`, answering `OPEN_QUESTIONS.md` item 12):
  the ARM side writes the instruction stream, kicks off execution, and polls
  a status register (or `stop_core`'s effect on it) rather than fielding an
  interrupt. This should be confirmed with the hardware author along with
  the other 12 open items -- it directly determines what `pipe_context::flush`
  and fence-wait will need to do once real hardware exists.

## 5. What this means for the Weeks 1-4 scope

Everything above steps 1 and "the compiler path" is explicitly deferred.
Weeks 1-4 build and prove: `pipe_screen`/`pipe_context` register with Mesa
under `GALLIUM_DRIVER=mgc`; a GLES2 context comes up over a surfaceless EGL
device (no PL fabric, no AXI, no kernel driver involved at all, since
nothing renders); `create_fs_state` receives real NIR and the backend
translates a basic-arithmetic subset of it into MGC assembly text; and that
text is verified by assembling and executing it in `tools/mgcsim`, entirely
on the host. The AXI/DMA link described above only becomes relevant once
there is a bitstream to talk to.
