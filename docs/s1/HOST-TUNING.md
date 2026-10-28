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

GPU clock: DPM level `high` while the GPU is busy, `auto` when idle - see
[Host settings (Strix Halo): GPU clock governor](#host-settings-strix-halo-gpu-clock-governor) below. (Until 2026-10-28 a
oneshot unit pinned `high` at boot for good.)

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

## Host settings (Strix Halo): GPU clock governor

`power_dpm_force_performance_level=high` keeps the iGPU at its top shader/memory clock instead of letting the default `auto`
governor ramp it per burst. On s1 (gfx1151, llama-server, paired off/on A/B 2026-10-28) `high` measured
**prefill +2.6 % at a 32k prompt and +3.2 % at 72k**; decode did not move in an earlier A/B (2026-09-07). Leaving `high` on all the
time costs about **+15-20 W at idle**, so I do not pin it any more: a tiny governor switches to `high` as soon as the GPU is
busy and back to `auto` after 120 s idle. The same script runs on balkon (780M); I have not measured the gain there.

`/usr/local/sbin/amdgpu-dpm-governor.sh` (picks the first amdgpu PCI device, or set `DEV=/sys/bus/pci/devices/<addr>`):

```sh
#!/bin/sh
# amdgpu DPM governor (piagent R138b, 2026-10-28): 'high' while the GPU is busy, 'auto' after IDLE_S seconds of idle.
# Installed on s1 (gfx1151, 0000:bd:00.0) and balkon (780M, 0000:c5:00.0) as /usr/local/sbin/amdgpu-dpm-governor.sh
# + scripts/host/amdgpu-dpm-governor.service. Polls gpu_busy_percent every POLL_S s; logs transitions to the journal.
# The level is re-read every poll, so a level forced from outside (scripts/bench/dpm-high.sh) also returns to auto
# after IDLE_S idle seconds. Measured on s1 (AB64 20261028-114936): high = prefill +2.6 % @32k / +3.2 % @72k;
# forcing high at idle costs ~ +15-20 W, hence the idle fallback.
# Rollback: systemctl disable --now amdgpu-dpm-governor; echo auto > $DEV/power_dpm_force_performance_level
if [ -z "${DEV:-}" ]; then
  for f in /sys/bus/pci/devices/*/power_dpm_force_performance_level; do [ -e "$f" ] && { DEV=${f%/*}; break; }; done
fi
CTL=${DEV:-/nonexistent}/power_dpm_force_performance_level; BUSY=${DEV:-/nonexistent}/gpu_busy_percent
POLL_S=${POLL_S:-1}; IDLE_S=${IDLE_S:-120}; BUSY_PCT=${BUSY_PCT:-10}
[ -w "$CTL" ] && [ -r "$BUSY" ] || { echo "amdgpu-dpm-governor: $CTL or $BUSY not usable" >&2; exit 1; }
idle=0
echo "amdgpu-dpm-governor: start on $DEV (current $(cat "$CTL"), busy>=${BUSY_PCT}% -> high, idle ${IDLE_S}s -> auto)"
while :; do
  b=$(cat "$BUSY" 2>/dev/null || echo 0); cur=$(cat "$CTL" 2>/dev/null || echo unknown)
  if [ "$b" -ge "$BUSY_PCT" ] 2>/dev/null; then
    idle=0
    if [ "$cur" != high ]; then echo high > "$CTL" && echo "amdgpu-dpm-governor: busy ${b}% -> high (was $cur)"; fi
  else
    idle=$((idle + POLL_S))
    if [ "$cur" != auto ] && [ "$idle" -ge "$IDLE_S" ]; then echo auto > "$CTL" && echo "amdgpu-dpm-governor: idle ${IDLE_S}s -> auto (was $cur)"; fi
  fi
  sleep "$POLL_S"
done
```

`/etc/systemd/system/amdgpu-dpm-governor.service`, then `sudo systemctl daemon-reload && sudo systemctl enable --now amdgpu-dpm-governor`:

```ini
[Unit]
Description=amdgpu DPM governor: high while busy, auto after idle (piagent R138b)
After=multi-user.target
[Service]
Type=simple
ExecStart=/usr/local/sbin/amdgpu-dpm-governor.sh
Restart=always
RestartSec=5
[Install]
WantedBy=multi-user.target
```

Check: `journalctl -u amdgpu-dpm-governor` shows the `start on …` line and a `busy N% -> high` line at the next GPU load;
`cat /sys/bus/pci/devices/*/power_dpm_force_performance_level` reads `high` under load and `auto` two minutes after it.

The 1 s poll reacts within a second, which is noise for a long prefill but not for a sub-second micro-benchmark. My benchmark
and test wrappers therefore also force `high` right before they start (one line, no failure if it cannot):

```sh
echo high | sudo -n tee /sys/bus/pci/devices/<addr>/power_dpm_force_performance_level >/dev/null
```

with a sudoers drop-in (`/etc/sudoers.d/power-dpm`, check it with `sudo visudo -cf` before installing) so it needs no password:

```
<user> ALL=(root) NOPASSWD: /usr/bin/tee /sys/bus/pci/devices/<addr>/power_dpm_force_performance_level
```

The governor re-reads the level every poll, so a level forced this way also returns to `auto` after the idle timeout.

Rollback: `sudo systemctl disable --now amdgpu-dpm-governor` and
`echo auto | sudo tee /sys/bus/pci/devices/<addr>/power_dpm_force_performance_level` (or `high` to pin it for good, as before).
If you keep an older boot-time "pin high" unit, the governor still works (it drops to `auto` after the first idle period),
but the unit is redundant.

## Docker

- `--device /dev/dri` and `group_add` with the host's `video` and `render` GIDs (`getent group video render`; s1 has 44/992,
  balkon 44/993).
- Vulkan only. ROCm needs `/dev/kfd` and `seccomp=unconfined` and is not what these images use.
- Keep model files on a fast local NVMe: the PLE table is read through the page cache, and the disk prompt cache reads at
  6.8 GB/s cold on my RAID 0.
