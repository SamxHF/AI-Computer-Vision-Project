# How To Run

This README only covers how to set up the environment and run the notebooks in this repo.

## 1. Create a virtual environment

From the project root:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

On Windows:

```bash
.venv\Scripts\activate
```

## 2. Install requirements

With the virtual environment activated, run:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## 3. Download the dataset

Download the DroneFace dataset from:

- Dataset page: `https://hjhsu.github.io/DroneFace/`
- Download link: `https://www.dropbox.com/s/3dvf4h6of4lf0rg/DroneFace.zip?dl=0`

After downloading, extract it into the project as `open_data_set/`.

## 4. Register the venv as a Jupyter kernel

This project must be run with the kernel from the virtual environment, not a global Python or Anaconda/base kernel.

Run:

```bash
python -m ipykernel install --user --name ai-computer-vision-project --display-name "Python (.venv)"
```

## 5. Open the notebooks

Open the repo in Jupyter, VS Code, or Cursor and use the `.venv` kernel for the notebook.

Notebooks in this repo:

- `people_identification_voting_simple.ipynb`
- `detector_recognizer_grid_search.ipynb`
- `face_classi.ipynb`

## 6. Make sure the notebook kernel is the venv kernel

Before running any cells, confirm the selected kernel is:

- `Python (.venv)` or
- the interpreter at `.venv/bin/python`

Do not use:

- a system Python kernel
- an Anaconda `base` kernel
- any kernel that is not connected to this repo's `.venv`

## 7. Run the notebook

After the correct kernel is selected, run the cells from top to bottom.
