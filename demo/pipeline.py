from motifs.config import PKG_DATA_PATH
from motifs.pipeline import Pipeline

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p",
        "--path",
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
    parser.add_argument("--no-plot", action="store_false")
    args = parser.parse_args()

    features = [{"name": "tfidf"}]
    n = 4
    pipeline = Pipeline(args.path, "motif", features, save=True)

    pipeline.execute(method=args.method, n=args.ngram, plot=False)
