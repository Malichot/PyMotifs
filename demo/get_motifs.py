from motifs.config import PKG_DATA_PATH
from motifs.pipeline import Pipeline

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--corpus_dir",
        type=str,
        help="Path to the corpus directory",
        default=PKG_DATA_PATH.joinpath("corpus_test"),
    )
    parser.add_argument(
        "-s",
        "--save",
        action="store_true",
        help="Save",
    )
    args = parser.parse_args()

    feature = {"name": "tfidf"}
    pipeline = Pipeline(
        "motif",
        feature,
        corpus_dir=args.corpus_dir,
        save=args.save,
    )
