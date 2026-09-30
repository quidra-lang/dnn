# Development and release workflow

This is the canonical release procedure for the `dnn` Quidra package.

## Permanent branches

- `main` is the latest published stable release source.
- `develop` is the long-lived integration branch for the next release.

Routine work goes directly to `develop`.
Do not use `main` for unreleased development. Do not delete and recreate
`develop` after a release.

## Release identity

A release is valid only when these agree:

- `project.toml` version `X.Y.Z`, equal to the coordinated Core/Math/NN
  version, and the `quidra.package` generated from it;
- immutable tag `vX.Y.Z`;
- the exact `main` commit carrying that manifest;
- the declared `requires.quidra` and package dependency ranges admitting
  same-version Core, Math, and NN;
- green CI against Core `vX.Y.Z`, Math `vX.Y.Z`, and NN `vX.Y.Z`.

Branches are never installation identities.

DNN uses the exact same `MAJOR.MINOR.PATCH` version as Core, Math, and NN.
It never chooses a release version independently.

## Releasing

When instructed to release dnn, release the library, or follow the release
procedure, execute the complete sequence:

1. Fetch the latest remote `develop` and `main` HEADs. Never work from a remembered SHA.
2. Confirm that all intended work is in `develop`.
3. Use the shared first-party release version selected for Core.
4. Update `project.toml`: set that exact shared version and the tested
   `requires.quidra`, `requires.math`, and `requires.nn` ranges. Then run
   `quidra package sync .` to regenerate `quidra.package`. Core owns package
   metadata parsing/generation; NN owns reusable neural-network backend assets.
5. Ensure the release workflow validates against immutable Core, Math, and NN
   tags with exactly the DNN package version. Development compatibility CI may
   additionally test the current `develop` branches.
6. Run/verify all tests and examples on `develop`. Fix failures there.
7. Merge `develop` into `main` while preserving valid history from both branches.
8. Let the release workflow triggered by the `main` push validate the immutable
   same-version Core/Math/NN tags, rerun DNN tests, refuse tag reuse, and create the
   immutable `vX.Y.Z` tag plus GitHub Release on that exact tested `main`
   commit. Do not pre-create or manually retarget the tag.
9. Verify the workflow succeeded and the tag, GitHub Release, and
   `quidra.package` version all match.
10. Return to `develop` and bring back any release-only change if needed.
    Change DNN's version only when the shared Core first-party version advances,
    then run `quidra package sync .` and push it.
11. Continue ordinary work only on `develop`.

Never force-move, delete/recreate, or reuse a published release tag.

## Backend ownership

DNN is the architecture/model composition layer and owns no reusable
accelerator runtime, package-native kernel, or generic neural-network compiler
policy. cuDNN/NCCL integration, NN-native kernels, execution policy and managed
NVIDIA assets belong to the lower-layer `nn` package. Generic BLAS/cuBLAS and
mathematical semantics belong to `math`. DNN releases therefore carry model
source only and depend on immutable NN/Math/Core releases.

## Dependency-first ordering

Every DNN release follows the same-version dependency tags in layer order:
Core `vX.Y.Z` first, then Math `vX.Y.Z`, then NN `vX.Y.Z`, and only then
DNN `vX.Y.Z`. The dependency ranges in `project.toml` must admit that shared
version, and release validation tests against those exact immutable tags.
