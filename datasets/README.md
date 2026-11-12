# Source graphs

Place the `.mat` files listed in the main README here. Graph files are not bundled.

Each file must contain an N-by-N adjacency matrix (`Network`, `A`, or `adj`) and an
N-by-F feature matrix (`Attributes`, `X`, or `features`). SciPy sparse matrices are
accepted. The complete field aliases are defined in `data/datasets.py`.

Optional original labels are retained as `raw_y`; they are separate from the
synthetic anomaly-type labels. Graphs are made undirected and feature normalization
is controlled by the dataset configuration. Use the same source graphs and
preprocessing as the experiment being reproduced.
