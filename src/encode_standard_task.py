"""Run the unchanged ordinary encoder with the launcher's DATA_ROOT."""
from pathlib import Path
import os


def main():
    from three_stage_retrieval_pipeline import _patch_dataset_paths
    _patch_dataset_paths(Path(os.environ.get("DATA_ROOT", "/root/autodl-tmp")))
    import encode
    return encode.main()


if __name__ == "__main__":
    main()
