"""CLI tool to inspect, configure, and switch AMD GPU hardware profiles for EdgeShield AI."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.amd_config import (
    DEFAULT_CONFIG_PATH,
    apply_amd_gpu_environment,
    load_amd_gpu_config,
    set_active_profile,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Configure AMD GPU and ROCm hardware execution profiles for EdgeShield AI."
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all configured AMD hardware profiles and current active selection",
    )
    parser.add_argument(
        "--set",
        dest="profile",
        type=str,
        help="Set active AMD profile (e.g. instinct_mi300, instinct_mi250, radeon_rx7900, radeon_rx6000, ryzen_ai_npu, auto)",
    )
    parser.add_argument(
        "--export-bash",
        action="store_true",
        help="Print bash export statements for current profile environment variables",
    )
    parser.add_argument(
        "--export-ps1",
        action="store_true",
        help="Print PowerShell $env: statements for current profile environment variables",
    )
    parser.add_argument(
        "--config-path",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to amd_gpu.yaml config file",
    )

    args = parser.parse_args()
    config = load_amd_gpu_config(args.config_path)

    if args.profile:
        print(f"[*] Setting active AMD GPU profile to: {args.profile}")
        try:
            config = set_active_profile(args.profile, args.config_path)
            print(f"[+] Successfully activated profile '{args.profile}'.")
        except ValueError as exc:
            print(f"[-] Error: {exc}", file=sys.stderr)
            return 1

    if args.export_bash:
        applied = apply_amd_gpu_environment(config_path=args.config_path)
        for k, v in applied.items():
            print(f"export {k}=\"{v}\"")
        return 0

    if args.export_ps1:
        applied = apply_amd_gpu_environment(config_path=args.config_path)
        for k, v in applied.items():
            print(f"$env:{k} = \"{v}\"")
        return 0

    # Default display
    print("=" * 68)
    print("EDGESHIELD AI — AMD GPU HARDWARE PROFILES")
    print("=" * 68)
    print(f"Active Profile: [{config.active_profile.upper()}]")
    print(f"Config File:    {args.config_path}")
    print("-" * 68)
    print(f"{'Profile Key':<18} | {'Architecture':<22} | {'HSA GFX':<10} | {'FP16':<5}")
    print("-" * 68)

    for key, p in config.profiles.items():
        is_active = " (ACTIVE)" if key == config.active_profile else ""
        print(
            f"{key + is_active:<18} | {p.architecture:<22} | {p.hsa_override_gfx_version or 'native':<10} | {str(p.half_precision):<5}"
        )

    print("=" * 68)
    print("\nApplied Environment Variables:")
    applied = apply_amd_gpu_environment(config_path=args.config_path)
    for k, v in applied.items():
        print(f"  {k} = \"{v}\"")

    print("\nUsage Tips:")
    print("  • Switch profile:    python scripts/configure_amd_gpu.py --set instinct_mi300")
    print("  • PowerShell export: python scripts/configure_amd_gpu.py --export-ps1 | Invoke-Expression")
    print("  • Linux bash export: eval $(python3 scripts/configure_amd_gpu.py --export-bash)")
    print("=" * 68)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
