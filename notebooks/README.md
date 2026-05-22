# Notebooks

Notebooks are version-controlled as `.py` (jupytext percent format).
To open one as a notebook:

    jupytext --to ipynb 01_eda.py
    jupyter lab notebooks/01_eda.ipynb

Edits to the `.ipynb` auto-sync back to `.py` on save.
Only commit the `.py`.