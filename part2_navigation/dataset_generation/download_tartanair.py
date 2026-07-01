"""
Flow-Constrained Video-to-Action Model for Cross-Embodiment Navigation

Author: Jeffrin Sam
Institution: Skoltech
Year: 2025
License: MIT

Description: Download TartanAir environments
"""

import argparse
from pathlib import Path
from typing import List


# Available TartanAir environments (corrected names)
AVAILABLE_ENVIRONMENTS = [
    "AbandonedFactory",
    "AbandonedFactory2", 
    "Hospital",
    "Office",
    "Neighborhood",  # Using ModularNeighborhood
    "Ocean",
    "OldTownSummer",
    "SeasideTown",
    "SeasonalForestSummer",
    "ConstructionSite",
    "Downtown",
    "JapaneseAlley",
    "Gascola",
    "SoulCity",
    "WesternDesertTown",
    "CarWelding",
    "EndofTheWorld"
]


def download_tartanair_env(env: str, difficulty: str, output_dir: Path,
                          modalities: List[str] = None, camera: str = "lcam_front"):
    """Download a TartanAir environment.

    Args:
        env: Environment name
        difficulty: Difficulty level (easy or hard)
        output_dir: Output directory
        modalities: Data modalities to download (image, depth, pose, etc.)
        camera: Camera name
    """
    if modalities is None:
        modalities = ['image', 'depth', 'pose']

    try:
        from tartanair import TartanAirDownloader

        # Initialize downloader with data root
        tartanair = TartanAirDownloader(tartanair_data_root=str(output_dir))

        print(f"\nDownloading {env} ({difficulty})...")
        print(f"  Modalities: {modalities}")
        print(f"  Camera: {camera}")
        print(f"  Output: {output_dir}")

        # Download
        tartanair.download(
            env=[env],
            difficulty=[difficulty],
            modality=modalities,
            camera_name=[camera],
            output_dir=str(output_dir)
        )

        print(f"Download complete: {env}")

    except ImportError:
        print("\nERROR: TartanAir library not installed!")
        print("Install with: pip install tartanair")
        return False

    except Exception as e:
        print(f"\nERROR downloading {env}: {e}")
        return False

    return True


def main():
    parser = argparse.ArgumentParser(description="Download TartanAir dataset environments")
    parser.add_argument("--env", type=str, nargs='+', default=None,
                       help="Environment name(s) to download")
    parser.add_argument("--all", action="store_true",
                       help="Download all available environments")
    parser.add_argument("--difficulty", type=str, nargs='+', default=['easy'],
                       choices=['easy', 'hard'],
                       help="Difficulty levels to download")
    parser.add_argument("--output_dir", type=str, default="data/tartanair",
                       help="Output directory for downloaded data")
    parser.add_argument("--modalities", type=str, nargs='+',
                       default=['image', 'depth', 'pose'],
                       help="Data modalities to download")
    parser.add_argument("--camera", type=str, default="lcam_front",
                       help="Camera name")
    parser.add_argument("--list", action="store_true",
                       help="List available environments and exit")

    args = parser.parse_args()

    # List environments and exit
    if args.list:
        print("\nAvailable TartanAir environments:")
        for i, env in enumerate(AVAILABLE_ENVIRONMENTS, 1):
            print(f"  {i:2d}. {env}")
        print(f"\nTotal: {len(AVAILABLE_ENVIRONMENTS)} environments")
        return

    # Determine which environments to download
    if args.all:
        environments = AVAILABLE_ENVIRONMENTS
    elif args.env is not None:
        environments = args.env
        # Validate environment names
        for env in environments:
            if env not in AVAILABLE_ENVIRONMENTS:
                print(f"WARNING: Unknown environment '{env}'")
                print(f"Available environments: {', '.join(AVAILABLE_ENVIRONMENTS)}")
    else:
        print("ERROR: Must specify either --env or --all")
        print("Use --list to see available environments")
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nTartanAir Download Configuration:")
    print(f"  Environments: {len(environments)}")
    print(f"  Difficulty levels: {args.difficulty}")
    print(f"  Modalities: {args.modalities}")
    print(f"  Camera: {args.camera}")
    print(f"  Output directory: {output_dir}")
    print(f"\nStarting download...\n")

    success_count = 0
    failed_envs = []

    for env in environments:
        for difficulty in args.difficulty:
            success = download_tartanair_env(
                env=env,
                difficulty=difficulty,
                output_dir=output_dir,
                modalities=args.modalities,
                camera=args.camera
            )

            if success:
                success_count += 1
            else:
                failed_envs.append(f"{env}/{difficulty}")

    print(f"\n{'='*60}")
    print(f"Download Summary:")
    print(f"  Successful: {success_count}")
    print(f"  Failed: {len(failed_envs)}")
    if failed_envs:
        print(f"  Failed environments:")
        for env in failed_envs:
            print(f"    - {env}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
