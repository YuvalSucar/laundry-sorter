from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parent
IMAGES_DIR = ROOT / 'תמונות'
IMAGES_DIR.mkdir(exist_ok=True)

# Only edit this mapping if your filenames are different
RENAME_MAP = {
    'IMG_7332.jpg': 'system_overview.jpg',
    'IMG_7331.jpg': 'robotic_arm.jpg',
    'צילום מסך 2026-04-15 ב-18.17.22.png': 'system_architecture.png',
    'צילום מסך 2026-04-15 ב-18.18.01.png': 'confusion_matrix.png',
    'צילום מסך 2026-04-15 ב-18.18.24.png': 'classification_report.png',
    'צילום מסך 2026-04-15 ב-18.22.26.png': 'dataset_sample.png',
}

IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png', '.webp'}


def safe_move(src: Path, dst: Path) -> None:
    if dst.exists():
        raise FileExistsError(f'Target file already exists: {dst.name}')
    shutil.move(str(src), str(dst))


def main() -> None:
    moved = []
    missing = []

    for old_name, new_name in RENAME_MAP.items():
        src = ROOT / old_name
        dst = IMAGES_DIR / new_name

        if not src.exists():
            missing.append(old_name)
            continue

        safe_move(src, dst)
        moved.append((old_name, dst.name))

    # Move any remaining image files into the folder without renaming them
    for item in ROOT.iterdir():
        if item.is_file() and item.suffix.lower() in IMAGE_SUFFIXES:
            dst = IMAGES_DIR / item.name
            if not dst.exists():
                safe_move(item, dst)
                moved.append((item.name, dst.name))

    print('\nFiles moved/renamed:')
    if moved:
        for old_name, new_name in moved:
            print(f'  {old_name} -> תמונות/{new_name}')
    else:
        print('  No files were moved.')

    print('\nMissing files (skip if they do not exist in your repo):')
    if missing:
        for name in missing:
            print(f'  {name}')
    else:
        print('  None')

    print('\nDone.')
    print('Now run: git add . && git commit -m "organize project files" && git push')


if __name__ == '__main__':
    main()
