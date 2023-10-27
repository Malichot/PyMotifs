from motifs.config import PKG_DATA_PATH
from motifs.features import build_specificity
from motifs.pipeline import Pipeline

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tokens_dir",
        type=str,
        help="Path to the tokens directory",
        default=None,
    )
    parser.add_argument(
        "--corpus_dir",
        type=str,
        help="Path to the corpus directory",
        default=PKG_DATA_PATH.joinpath("corpus_test"),
    )
    parser.add_argument(
        "-m",
        "--method",
        type=str,
        help="Method for features analysis",
        default="pca",
    )
    parser.add_argument(
        "-n", "--ngram", type=int, help="n-gram length", default=4
    )
    parser.add_argument(
        "-s",
        "--save",
        action="store_true",
        help="Save",
    )
    parser.add_argument("--no-plot", action="store_false")
    args = parser.parse_args()

    features = [{"name": "tfidf"}]
    pipeline = Pipeline(
        "motif",
        features,
        tokens_dir=args.tokens_dir,
        corpus_dir=args.corpus_dir,
        save=args.save,
    )

    # Compute features
    pipeline.execute(method=args.method, n=args.ngram, plot=False)

    # Specificity
    spec = build_specificity(pipeline.ngrams)
    print(spec)
