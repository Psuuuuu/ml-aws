import os
from pathlib import Path
import sys


EXCLUDED_DIRS = {"venv", ".venv", "__pycache__", ".git"}
ALLOWED_EXTENSIONS = {".py", ".csv", ".yaml", ".yml"}
ALLOWED_FILENAMES = {".env"}


def should_exclude_dir(dirname: str) -> bool:
    return dirname in EXCLUDED_DIRS


def should_include_file(path: Path, root: Path, output_path: Path) -> bool:
    # Exclude the running script itself
    if path.resolve() == Path(sys.argv[0]).resolve():
        return False

    # Exclude the generated markdown output
    if path.resolve() == output_path.resolve():
        return False

    if path.name in ALLOWED_FILENAMES:
        return True

    if path.suffix.lower() in ALLOWED_EXTENSIONS:
        return True

    return False


def build_tree(root: Path) -> str:
    lines = []

    def _tree(dir_path: Path, prefix: str = ""):
        entries = sorted(
            [
                p
                for p in dir_path.iterdir()
                if not (p.is_dir() and should_exclude_dir(p.name))
            ]
        )

        for index, entry in enumerate(entries):
            connector = "└── " if index == len(entries) - 1 else "├── "
            lines.append(prefix + connector + entry.name)

            if entry.is_dir():
                extension = "    " if index == len(entries) - 1 else "│   "
                _tree(entry, prefix + extension)

    lines.append(root.name)
    _tree(root)
    return "\n".join(lines)


def read_file_content(path: Path) -> str:
    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            lines = []
            for _ in range(2):
                line = f.readline()
                if not line:
                    break
                lines.append(line.rstrip("\n"))
        return "\n".join(lines)

    with path.open("r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def collect_files(root: Path, output_path: Path):
    files = []

    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if not should_exclude_dir(d)]

        for filename in filenames:
            path = Path(dirpath) / filename

            if should_include_file(path, root, output_path):
                files.append(path)

    return sorted(files)


def write_markdown(root: Path, output_file: Path):
    tree_text = build_tree(root)
    files = collect_files(root, output_file)

    with output_file.open("w", encoding="utf-8") as md:
        md.write("# Project Structure\n\n")
        md.write("```text\n")
        md.write(tree_text)
        md.write("\n```\n\n")

        md.write("# File Contents\n\n")

        for file_path in files:
            relative_path = file_path.relative_to(root)
            content = read_file_content(file_path)

            md.write(f"## {relative_path}\n\n")
            md.write("```\n")
            md.write(content)
            md.write("\n```\n\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--project_path", type=str, default="./")
    parser.add_argument("--output", type=str, default="project_dump.md")

    args = parser.parse_args()

    root_path = Path(args.project_path).resolve()
    output_path = Path(args.output).resolve()

    write_markdown(root_path, output_path)
    print(f"Markdown file created at: {output_path}")
