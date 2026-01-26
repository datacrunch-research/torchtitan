# Docker Build Fixes for PyTorch Nightly

This document tracks compatibility fixes applied for running torchtitan with PyTorch nightly in Docker.

## Fixed Issues

### 1. Fake Backend Compatibility (RESOLVED)

**Error:**
```
RuntimeError: Backend fake does not yet support sequence numbers.
```

**Root Cause:**
PyTorch nightly removed support for sequence numbers in the fake backend, but torchtitan's `parallel_dims.py` was using fake backend for process groups with degree 1.

**Fix Applied:**
Modified `/root/torchtitan/torchtitan/distributed/parallel_dims.py` (line 108-125) to disable fake backend usage:
- Commented out the `backend_override` logic that sets backend to "fake"
- Process groups are now created with the default backend (NCCL for CUDA)

**File:** [torchtitan/distributed/parallel_dims.py](torchtitan/distributed/parallel_dims.py:108-125)

### 2. Requirements File Path (RESOLVED)

**Error:**
```
ERROR: Could not open requirements file: [Errno 2] No such file or directory: 'requirements.txt'
```

**Root Cause:**
`requirements.txt` is a symlink to `.ci/docker/requirements.txt`, but `.dockerignore` was excluding the `.ci/` directory.

**Fix Applied:**
Modified `.dockerignore` to allow `.ci/docker/` directory and `.txt` files:
```dockerignore
.ci/*
!.ci/docker/
!.ci/docker/*.txt
```


## Notes

- These fixes are specific to PyTorch nightly compatibility
- The fake backend was an optimization to avoid creating unnecessary process groups
- Removing it may slightly increase overhead for single-degree dimensions, but ensures compatibility
- If you need to revert these changes, uncomment the backend_override logic in parallel_dims.py
