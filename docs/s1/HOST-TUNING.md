# Host tuning (my setup, adapt)

Both boxes run Ubuntu with kernel 7.0 (`7.0.0-34-generic`), Mesa RADV for Vulkan, Docker with `/dev/dri` passed in. These are
the lines actually in use on 2026-10-10, read from `/proc/cmdline` and `/etc/sysctl.d`. Items marked *(my setup)* are there for
reasons that are not specific to LLM serving or that I have not isolated - check them against your own needs.

## s1: Ryzen AI Max+ 395 / Radeon 8060S (gfx1151), 128 GB LPDDR5x

BIOS: the dedicated VRAM carve-out is set to the minimum (512 MB). Everything the GPU uses comes from GTT (system RAM mapped for
the GPU), which is what the kernel parameters below size. The OS sees ~124 GiB.

Kernel command line (`GRUB_CMDLINE_LINUX_DEFAULT` in `/etc/default/grub`, then `sudo update-grub` and reboot):

```
ttm.pages_limit=28835840 ttm.page_pool_size=28835840   # 28 835 840 x 4 KiB pages = 110 GiB (112 640 MiB) of GTT
amd_iommu=off                                           # measured +13 % decode, +25 % prefill, +56 % 4-stream vs iommu=pt
amdgpu.lockup_timeout=600000                            # long prompts: with the default 10 s Vulkan died at ~155k prompt tokens
```

- GTT size: I set it with `ttm.pages_limit` / `ttm.page_pool_size`, not `amdgpu.gttsize`. Check after reboot:
  `cat /sys/class/drm/card*/device/mem_info_gtt_total` (should read 118 111 600 640 bytes = 110 GiB).
- 110 GiB is needed only for the 1M profile (102.7 GiB GTT). The 524k profile (92.3 GiB) ran for weeks with 104 GiB
  (`ttm.pages_limit=27262976`). Leaving ~14 GiB to the OS is enough only because nothing else big runs on the box.
- *(my setup)* also on the command line: `nvme_core.default_ps_max_latency_us=0 pcie_aspm=off pcie_port_pm=off
  processor.max_cstate=1 mitigations=off transparent_hugepage=always` and a ramoops reservation. These came from chasing NVMe
  power-state issues and crash logging on this unit, not from LLM tuning; I have not measured their effect on inference.

Memory sysctls (`/etc/sysctl.d/90-s1-memory.conf`):

```
vm.dirty_bytes = 2147483648            # cap dirty page cache at 2 GiB: writing a 100 GB GGUF next to a loaded model
vm.dirty_background_bytes = 536870912  # otherwise fills RAM with dirty pages and stalls the box
vm.min_free_kbytes = 2097152           # keep 2 GiB free so the kernel can reclaim before the GPU allocation fails
vm.watermark_scale_factor = 200
vm.swappiness = 10
```

Note: another file in `/etc/sysctl.d` that sorts later (mine was `99-perf.conf`) can silently override the dirty caps with
`vm.dirty_ratio` / `vm.dirty_background_ratio`. Check with `sysctl vm.dirty_bytes vm.dirty_ratio`.

Swap: 64 GiB of swap files. The model never swaps; swap is there so a one-off job (a 32 GB logits dump once) does not take the
box down.

GPU clock: a oneshot systemd unit pins the DPM level to high at boot (+3 % prefill, decode unchanged):

```
echo high > /sys/class/drm/card1/device/power_dpm_force_performance_level     # card index may differ
```

Resilience *(my setup, worth copying)*:
- `sshd`: `OOMScoreAdjust=-1000`, `MemoryLow=1G` in a drop-in, so an out-of-memory event never takes SSH with it.
- systemd-oomd kill policy only on a dedicated slice for batch jobs; never on the user session slice (it killed my editor and
  SSH sessions twice).
- Hardware watchdog armed (`RuntimeWatchdogSec=60` in `/etc/systemd/system.conf`, `sp5100_tco`): a hard hang resets the box.
- One heavy GPU or RAM job at a time. Two model loads at once, or a GGUF conversion next to a model reload, hung or OOM'd the box
  more than once. I serialise them with one `flock`.

Cooling: a 1M cold fill keeps the GPU at full load for over an hour, and a busy agent day is hours of decode. My unit runs
in its default power profile; I have not measured throttling. Check temperatures under a long fill before you trust numbers.

## balkon: Ryzen 9 7940HS / Radeon 780M (gfx1103, RDNA3), 96 GB DDR5

The 780M has 12 CUs and ~80 GB/s effective memory bandwidth from the iGPU, so it is the slower "hands" box.

Kernel command line:

```
ttm.pages_limit=20447232 ttm.page_pool_size=20447232   # 78 GiB of GTT (out of ~93 GiB visible)
amd_iommu=off
amdgpu.lockup_timeout=600000
```

Sysctls (`/etc/sysctl.d/90-*.conf`), s1's file scaled down:

```
vm.dirty_bytes = 1073741824
vm.dirty_background_bytes = 268435456
vm.min_free_kbytes = 1048576
vm.watermark_scale_factor = 200
vm.swappiness = 10
```

Swap: 128 GiB (a 96 GiB file + a 32 GiB partition) *(my setup; more than needed)*.

Threads: use the 8 physical cores (`-t 8`); 16 threads with SMT measured slower.

## Docker

- `--device /dev/dri` and `group_add` with the host's `video` and `render` GIDs (`getent group video render`; s1 has 44/992,
  balkon 44/993).
- Vulkan only. ROCm needs `/dev/kfd` and `seccomp=unconfined` and is not what these images use.
- Keep model files on a fast local NVMe: the PLE table is read through the page cache, and the disk prompt cache reads at
  6.8 GB/s cold on my RAID 0.
