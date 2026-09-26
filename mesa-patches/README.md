# mesa-patches/

`mesa/` (the upstream Mesa 25.2.0 checkout) is untracked by this repo's
git (see `.gitignore`) — it's 372 MB of someone else's source tree, not
ours to version. Our actual changes to it — the `mgc` Gallium driver — are
committed on a branch *inside* `mesa/`'s own git history (branch
`mgc-driver`, based on tag `mesa-25.2.0`) and exported here as a patch
series, so the diff is reviewable from this repo without carrying Mesa
itself.

## Applying

```
cd mesa
git am ../mesa-patches/*.patch
```

(`mesa/` must already be at the `mesa-25.2.0` tag, detached HEAD, clean.)

## Regenerating after further driver work

From inside `mesa/`, on the `mgc-driver` branch:

```
git format-patch mesa-25.2.0 -o ../mesa-patches/
```

See `docs/driver/gallium-integration.md` and the Weeks 1-4 plan for what
the driver does and why it's structured this way.
