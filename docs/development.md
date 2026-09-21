# Development and release workflow

This is the canonical release procedure for the `dnn` Quidra package.

## Permanent branches

- `main` is the latest published stable release source.
- `develop` is the long-lived integration branch for the next release.

Normal work goes to `develop` (directly or through temporary branches).
Do not use `main` for unreleased development. Do not delete and recreate
`develop` after a release.

## Release identity

A release is valid only when these agree:

- `quidra.package` version `X.Y.Z`;
- immutable tag `vX.Y.Z`;
- the exact `main` commit carrying that manifest;
- the declared `requires.quidra` and package dependency ranges;
- green CI against the released Quidra version represented by that range.

Branches are never installation identities.

## Releasing

When instructed to release dnn, release the library, or follow the release
procedure, execute the complete sequence:

1. Fetch the latest remote `develop` and `main` HEADs. Never work from a remembered SHA.
2. Confirm that all intended work is in `develop`.
3. Choose the Semantic Versioning release number from the actual change.
4. Update `quidra.package`: set the exact package version, the tested
   `requires.quidra` range, and any `requires.<package>` ranges.
5. Ensure CI tests against an immutable released Quidra tag, never a Quidra
   development branch.
6. Run/verify all tests and examples on `develop`. Fix failures there.
7. Merge `develop` into `main` while preserving valid history from both branches.
8. Create immutable tag `vX.Y.Z` on that exact tested `main` commit and create
   the GitHub Release. Never tag `develop`.
9. Verify the tag, GitHub Release, and `quidra.package` version all match.
10. Return to `develop`, bring back any release-only change if needed, advance
    `quidra.package` to the next intended development version, and push it.
11. Continue ordinary work only on `develop`.

Never force-move, delete/recreate, or reuse a published release tag.

## Accelerator component identity

Native accelerator components belong to DNN releases. If a release includes
cuDNN, cuBLAS, NCCL, Metal-specific support, or ROCm/HIP user-space components,
record and package their exact versions, artifact identities, and checksums as
part of that immutable DNN release. The same DNN version on the same supported
OS/architecture must resolve to the same component set. Do not use a mutable
"latest" component or silently prefer a system-installed CUDA/cuDNN stack.

## Core-first ordering

If dnn needs a newer Quidra core, release Quidra core first. Only after that
immutable core tag exists should dnn update `requires.quidra`, test against
that tag, and publish its own release.
