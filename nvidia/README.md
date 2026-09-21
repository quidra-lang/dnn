# NVIDIA managed bundle contract

This directory is the package-owned location for an optional managed NVIDIA DNN
runtime bundle. Quidra Core owns CUDA driver/device execution; the `dnn`
package owns the cuBLAS, cuDNN, and NCCL release policy.

## Installed layout

A populated installed package uses:

```text
~/.quidra/packages/dnn/
  nvidia/
    BUNDLE.json
    SHA256SUMS
    lib/
      <cuBLAS shared libraries>
      <cuDNN shared libraries>
      <NCCL shared libraries>
```

On Windows the equivalent root is
`%USERPROFILE%\.quidra\packages\dnn\nvidia\lib`.

The Core runtime probes `nvidia/lib` automatically. It does not fall back to
system cuBLAS/cuDNN/NCCL unless
`QUIDRA_DNN_ALLOW_SYSTEM_NVIDIA_LIBS=1` is set. An explicit
`QUIDRA_DNN_NVIDIA_LIBRARY_PATH` takes precedence and is strict.

## Metadata schema

`BUNDLE.json` uses schema version 1. The canonical shape is shown in
`BUNDLE.example.json`. It must contain the exact DNN release version,
`platform`, `architecture`, a non-empty `cuda_compatibility` description,
and the components supported on that platform. Linux requires `cublas`,
`cudnn`, and `nccl`; Windows requires `cublas` and `cudnn` because Core does
not expose the NCCL backend on Windows. Each component records both an exact
`version` and an immutable `artifact` identity. The validator rejects partial metadata, extra/missing components,
unsafe checksum paths, symlinks, checksum omissions, extra checksum entries, and
checksum mismatches.

## Release contract

A managed bundle is release data, not an unversioned system dependency. Before a
DNN release publishes a populated bundle:

1. `BUNDLE.json` must identify the DNN release, supported platform/architecture,
   CUDA compatibility, and exact versions for every component required by the
   selected platform. Each component must also list its non-empty `files`
   inventory under `lib/`;
   inventories may not overlap and together must cover the bundle exactly.
2. Every file under `lib/` must be covered by `SHA256SUMS`.
3. The checksums must be computed from the exact artifacts shipped with the
   release; floating URLs or "latest" aliases are not acceptable.
4. The bundle must be validated on the same released Quidra Core baseline named
   by `requires.quidra`.
5. Replacing a published bundle requires a new DNN release; published checksums
   are immutable.

The repository intentionally does not contain vendor binaries during ordinary
development until a release has selected redistributable artifacts and recorded
their exact versions and checksums. Normal CI therefore permits an unstaged
bundle contract, but the release workflow runs the validator with
`--require-populated` and refuses to create a DNN release while the managed
bundle is empty. This prevents the managed-bundle path from silently becoming a
system-library search path or an unpinned download channel.


## Selected release artifacts

The immutable source selection for the managed bundle is recorded in
`SOURCES.json`. The selected CUDA family is 13. Linux x86_64 uses cuBLAS
13.8.0.4, cuDNN 9.26.0.51 (CUDA 13), and NCCL 2.31.2. Windows x86_64 uses
cuBLAS 13.8.0.4 and cuDNN 9.26.0.51 (CUDA 13). Every source artifact is pinned
by exact identity and SHA256; floating "latest" URLs are forbidden.

Vendor archives remain outside Git history. Bundle materialization must verify
the source SHA256 before extracting libraries into `nvidia/lib`, preserve the
vendor license files, then generate `BUNDLE.json` and `SHA256SUMS` for the
exact files that will be installed.

The package manifest declares the immutable Linux x86_64 and Windows
x86_64 GitHub release assets. A Quidra Core within the declared compatibility
range downloads the matching asset
during `quidra install dnn`, validates the archive structure before extraction,
and atomically publishes the source package together with its managed NVIDIA
runtime. macOS installs no NVIDIA asset and continues to use the Metal backend.
