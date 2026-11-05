# Data

Large datasets are **not committed** to Git (see `.gitignore`). Follow one of the options below.

## Dataset: AG News Classification (Kaggle)

* Kaggle page: <https://www.kaggle.com/datasets/amananandrai/ag-news-classification-dataset>
* `train.csv` - 120,000 articles, `test.csv` - 7,600 articles, perfectly balanced classes
* Columns: `Class Index`, `Title`, `Description`

| Class Index | Category |
|:---:|---|
| 1 | World |
| 2 | Sports |
| 3 | Business |
| 4 | Sci/Tech |

### Option A - manual download
1. Open the Kaggle link and click **Download**.
2. Unzip and place **both** files here:
   ```
   data/raw/train.csv
   data/raw/test.csv
   ```

### Option B - Kaggle CLI
```bash
pip install kaggle                       # put your API token in ~/.kaggle/kaggle.json
kaggle datasets download -d amananandrai/ag-news-classification-dataset -p data/raw --unzip
```

Then run `python -m news_classification.train`.

## How the files are used

| File | Role |
|---|---|
| `train.csv` | Split 85 % **train** / 15 % **validation** (stratified). Models are compared and the best one is chosen on the validation set. |
| `test.csv` | **Official held-out test set**, used only for the final report. |

If `test.csv` is missing, the pipeline automatically holds out a stratified 20 % test split from `train.csv`.
Any training article that also appears in the test file is removed to prevent leakage.

## Using another dataset

Any CSV with a label column (values 1-4 or 0-3) and a text column works:

```bash
python -m news_classification.train --data my_train.csv --test-data my_test.csv \
    --label-col label --title-col "" --text-col text
```
(`--title-col ""` means the file has no separate title column.) The four category names are
defined in `src/news_classification/config.py` (`CLASS_NAMES`); edit them for other label sets.

## Bundled sample (`data/sample/`)

`sample_train.csv` (480 rows) and `sample_test.csv` (120 rows) are **synthetic**, template-generated
articles in the AG News format, created by `scripts/generate_sample_data.py`. They exist only for unit
tests, CI and a quick smoke run (`make train-sample`). **Never report results obtained on them.**

## Folders

| Path | Purpose | Tracked by Git |
|---|---|---|
| `data/raw/` | Original Kaggle downloads | no |
| `data/processed/` | `clean_train.csv` written by the pipeline | no |
| `data/sample/` | Synthetic smoke-test data | yes |
