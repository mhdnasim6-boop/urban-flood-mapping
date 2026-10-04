"""Generate the Kaggle notebooks in this folder (run after editing).

    python kaggle/make_notebooks.py

Every notebook clones the GitHub repo, prints the commit it runs, and checks its
inputs before doing any work, so a missing input fails in seconds instead of
silently recomputing hours of work.
"""
import json
from pathlib import Path

HERE = Path(__file__).parent

SETUP = r'''
# ---- Settings -------------------------------------------------------------
GITHUB_REPO = "https://github.com/mhdnasim6-boop/urban-flood-mapping.git"
BRANCH = "main"
# ---------------------------------------------------------------------------
# Private repo: add a Kaggle secret named GITHUB_TOKEN (Add-ons > Secrets).
# Optional: a secret named HF_TOKEN (Hugging Face read token) avoids download rate limits.
import glob, os, subprocess
from pathlib import Path

def secret(name):
    try:
        from kaggle_secrets import UserSecretsClient
        return UserSecretsClient().get_secret(name)
    except Exception:
        return None

url = GITHUB_REPO
if secret("GITHUB_TOKEN"):
    url = url.replace("https://", f"https://{secret('GITHUB_TOKEN')}@")
if secret("HF_TOKEN"):
    os.environ["HF_TOKEN"] = secret("HF_TOKEN")
subprocess.run(["rm", "-rf", "/tmp/ufm"])
subprocess.run(["git", "clone", "-q", "--depth", "1", "-b", BRANCH, url, "/tmp/ufm"], check=True)
os.chdir("/tmp/ufm")
print("code version:", subprocess.run(["git", "log", "-1", "--oneline"], capture_output=True, text=True).stdout)

def find_input(name, notebook):
    """Locate an attached input folder or stop immediately with a clear message."""
    hits = sorted(glob.glob(f"/kaggle/input/**/{name}", recursive=True))
    if not hits:
        raise SystemExit(f"STOP: input '{name}' not found. Add the output of notebook {notebook} "
                         "(Add Input > Your Work > Notebooks) and run again.")
    print(f"{name}: {hits[0]}")
    return hits[0]
'''

PIP = "!pip install -q segmentation-models-pytorch tabulate rasterio 2>/dev/null; python -c \"import rasterio, segmentation_models_pytorch as s; print('ok', s.__version__)\""


def cell(kind, src):
    src = src.strip("\n")
    lines = [l + "\n" for l in src.split("\n")]
    lines[-1] = lines[-1].rstrip("\n")
    c = {"cell_type": kind, "metadata": {}, "source": lines}
    if kind == "code":
        c.update({"execution_count": None, "outputs": []})
    return c


def notebook(cells):
    return {"cells": [cell(k, s) for k, s in cells],
            "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                         "language_info": {"name": "python"}},
            "nbformat": 4, "nbformat_minor": 5}


def header(title, accel, inputs, output, time, body):
    inp = "\n".join(f"{i}. {x}" for i, x in enumerate(inputs, 1)) or "none"
    return f"""# {title}

**Settings:** Accelerator = **{accel}**, Internet = On. Run with **Save Version > Save & Run All (Commit)**,
not the editor's Run All: a committed run keeps going when the laptop is closed and its output is saved.

**Inputs** (Add Input > Your Work > Notebooks):
{inp}

**Output:** {output}  |  **Approx. time:** {time}

{body}"""


def train_cells(title, jobs, mc, time, body):
    return [
        ("markdown", header(title, "GPU T4 x2", ["`01_prepare_trainval` (ufm-trainval)", "`02_prepare_test` (ufm-test)"],
                            "`runs/<experiment>/` (checkpoints, metrics, maps)", time, body)),
        ("code", SETUP),
        ("code", f"""TRAINVAL = find_input("ufm-trainval", "01_prepare_trainval")
TEST = find_input("ufm-test", "02_prepare_test")
JOBS = {jobs!r}      # "<experiment>@<k>" = extra random seed k
MC = {mc!r}"""),
        ("code", PIP + "\n!nvidia-smi --query-gpu=name,memory.total --format=csv"),
        ("code", """# trains two experiments at a time (one per T4); progress lines are prefixed [gpu0]/[gpu1]
!python scripts/run_queue.py {' '.join(JOBS)} --trainval-root {TRAINVAL} --test-root {TEST} --out /kaggle/working/runs --mc {' '.join(MC) or '""'}"""),
        ("code", "!for f in /kaggle/working/runs/*/test/test_metrics.csv; do echo \"== $f\"; grep ALL $f | cut -c1-120; done"),
    ]


NOTEBOOKS = {
    "01_prepare_trainval.ipynb": [
        ("markdown", header("01 - Prepare training/validation data", "None (CPU)", [],
                            "`ufm-trainval` (~10 GB)", "1-1.5 h",
                            "Streams the 38 GB UrbanSARFloods archive (never stored), converts every chip to compact "
                            "uint8 GeoTIFFs, adds HAND / slope / permanent-water layers and computes normalisation "
                            "statistics. Expected: 6471 chips (4770 NF, 1011 FO, 690 FU) and class pixel counts "
                            "[1161425968, 18120626, 363550].")),
        ("code", SETUP),
        ("code", "!pip install -q rasterio 2>/dev/null; python -c \"import rasterio; print('rasterio', rasterio.__version__)\""),
        ("code", "!python scripts/prepare_train.py --out /kaggle/working/ufm-trainval"),
        ("code", "!python scripts/build_aux.py --root /kaggle/working/ufm-trainval --cache /tmp/tiles --workers 4 --max-cache-gb 6"),
        ("code", "!python scripts/compute_stats.py --root /kaggle/working/ufm-trainval"),
        ("code", """!rm -f /kaggle/working/ufm-trainval/_stable_samples.npz
!du -sh /kaggle/working/ufm-trainval
!for d in /kaggle/working/ufm-trainval/*/*/; do echo "$d $(ls $d | wc -l)"; done"""),
    ],
    "02_prepare_test.ipynb": [
        ("markdown", header("02 - Prepare the held-out test events", "None (CPU)", [],
                            "`ufm-test` (~3.3 GB)", "0.5-1 h",
                            "Downloads the test chips of Weihui (China), Nova Kakhovka (Ukraine) and Jubba (Somalia, "
                            "two image pairs) plus their full-frame reference maps, and adds the auxiliary layers. "
                            "Expected chips: 2475 / 2893 / 1779 / 1032 (997 empty Jubba_2 files on the server are skipped).")),
        ("code", SETUP),
        ("code", "!pip install -q rasterio 2>/dev/null; python -c \"import rasterio; print('rasterio', rasterio.__version__)\""),
        ("code", "!python scripts/prepare_test.py --out /kaggle/working/ufm-test --threads 16"),
        ("code", "!python scripts/build_aux.py --root /kaggle/working/ufm-test --cache /tmp/tiles --workers 4 --max-cache-gb 6"),
        ("code", "!du -sh /kaggle/working/ufm-test/*\n!cat /kaggle/working/ufm-test/event_stats.json"),
    ],
    "03_train_core.ipynb": train_cells(
        "03 - Train the core models (proposed, baselines, seeds)",
        ["D_full", "B5_authors_baseline", "A_raw", "D_full@1", "A_raw@1", "D_full@2", "A_raw@2"], ["D_full"],
        "4-6 h",
        "Proposed model D, the dataset authors' setup B5, the 8-band model A, and two extra random seeds of A and D "
        "(mean +- sd). D also gets 20-pass MC-dropout uncertainty maps."),
    "04_train_ablation.ipynb": train_cells(
        "04 - Train the ablation models",
        ["B_raw_phys", "C_raw_aux", "E_no_coherence", "F_no_vh"], [],
        "2-3 h",
        "B = + physics features, C = + terrain/water, E = without coherence, F = without VH. "
        "Can run at the same time as notebook 03 if Kaggle allows two GPU sessions; otherwise run it after."),
    "05_baselines.ipynb": [
        ("markdown", header("05 - Comparison methods B1-B4", "None (CPU)",
                            ["`01_prepare_trainval` (ufm-trainval)", "`02_prepare_test` (ufm-test)"],
                            "`runs/baselines/`", "1.5-3 h (no GPU quota used)",
                            "B1 intensity change (Otsu), B2 coherence threshold 0.3, B3 rule-based tree, "
                            "B4 random forest with validation-tuned class weights; all on 5x5 median-filtered SAR.")),
        ("code", SETUP),
        ("code", """TRAINVAL = find_input("ufm-trainval", "01_prepare_trainval")
TEST = find_input("ufm-test", "02_prepare_test")"""),
        ("code", "!pip install -q rasterio tabulate 2>/dev/null; python -c \"import cv2, xgboost; print('cv2', cv2.__version__, 'xgboost', xgboost.__version__)\""),
        ("code", "!python scripts/run_baselines.py --test-root {TEST} --trainval-root {TRAINVAL} --out /kaggle/working/runs/baselines --rf-device cpu\n!cat /kaggle/working/runs/baselines/settings.json"),
    ],
    "06_report.ipynb": [
        ("markdown", header("06 - Comparison report and uncertainty analysis", "None (CPU)",
                            ["`02_prepare_test` (ufm-test)", "`03_train_core`", "`04_train_ablation`", "`05_baselines`"],
                            "`report/` (tables, bar chart, uncertainty, maps)", "15-30 min",
                            "Collects every run from the attached notebooks, checks nothing is missing, then writes the "
                            "comparison table (seed mean +- sd, Jubba_2 excluded), the uncertainty analysis and the maps.")),
        ("code", SETUP),
        ("code", """TEST = find_input("ufm-test", "02_prepare_test")
EXCLUDE = ["20231201_Jubba_2"]   # too few reference flood pixels inside the published chips
EXPECTED = ["D_full", "B5_authors_baseline", "A_raw", "D_full_s1", "A_raw_s1", "D_full_s2", "A_raw_s2",
            "B_raw_phys", "C_raw_aux", "E_no_coherence", "F_no_vh",
            "baselines/B1", "baselines/B2", "baselines/B3", "baselines/B4"]

# link every run folder from the attached notebooks into one place (nothing is copied)
RUNS = Path("/tmp/runs"); (RUNS / "baselines").mkdir(parents=True, exist_ok=True)
for src in glob.glob("/kaggle/input/**/runs", recursive=True):
    for d in Path(src).iterdir():
        if d.name == "baselines":
            for b in d.iterdir():
                if not (RUNS / "baselines" / b.name).exists():
                    (RUNS / "baselines" / b.name).symlink_to(b)
        elif d.is_dir() and not (RUNS / d.name).exists():
            (RUNS / d.name).symlink_to(d)
missing = [e for e in EXPECTED if not (RUNS / e / "test" / "test_metrics.csv").exists()]
if not (RUNS / "D_full" / "test_mc" / "test_metrics.csv").exists():
    missing.append("D_full/test_mc (MC-dropout maps)")
if missing:
    raise SystemExit(f"STOP: missing runs {missing}. Attach notebooks 03, 04 and 05 (after they finish).")
print("all runs present:", sorted(p.name for p in RUNS.iterdir()))"""),
        ("code", "!pip install -q rasterio tabulate 2>/dev/null"),
        ("code", "!python scripts/uncertainty_analysis.py --maps {RUNS}/D_full/test_mc --test-root {TEST} --out /kaggle/working/report --exclude-events {' '.join(EXCLUDE)}"),
        ("code", "!python scripts/summarize.py --runs {RUNS} --test-root {TEST} --out /kaggle/working/report --exclude-events {' '.join(EXCLUDE)}"),
        ("code", """!cp /tmp/runs/baselines/settings.json /kaggle/working/report/baseline_settings.json 2>/dev/null; true
!for r in /tmp/runs/*/history.csv; do cp $r /kaggle/working/report/history_$(basename $(dirname $r)).csv; done
from IPython.display import Markdown, Image, display
for md in ["comparison.md", "uncertainty.md"]:
    display(Markdown(open(f"/kaggle/working/report/{md}").read()))
for p in ["f1_bars.png", "uncertainty.png"] + sorted(p.name for p in Path("/kaggle/working/report").glob("maps_*.png")):
    display(Image(f"/kaggle/working/report/{p}"))"""),
    ],
}

if __name__ == "__main__":
    for old in HERE.glob("*.ipynb"):
        if old.name not in NOTEBOOKS:
            old.unlink()
            print("removed", old.name)
    for name, cells in NOTEBOOKS.items():
        (HERE / name).write_text(json.dumps(notebook(cells), indent=1))
        print("wrote", name)
