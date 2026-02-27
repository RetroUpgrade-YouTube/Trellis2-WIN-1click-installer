from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
INPUT_DIR = BASE_DIR / "input"
OUTPUT_DIR = BASE_DIR / "outputs"

MODELS_DIR.mkdir(exist_ok=True)
INPUT_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# ---- API expected by TRELLIS ----

models_dir = str(MODELS_DIR)


def get_input_directory():
    return str(INPUT_DIR)


def get_output_directory():
    return str(OUTPUT_DIR)


def get_annotated_filepath(filename):
    return str(INPUT_DIR / filename)


def exists_annotated_filepath(filename):
    return (INPUT_DIR / filename).exists()


def get_save_image_path(prefix, output_dir):
    """
    Mimics ComfyUI behavior enough for TRELLIS exports
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    filename = Path(prefix).stem
    counter = 0
    subfolder = ""

    return (
        str(output_dir),
        filename,
        counter,
        subfolder,
        prefix,
    )
