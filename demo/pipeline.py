from motifs.config import PKG_DATA_PATH
from motifs.pipeline import Pipeline

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
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

    CORPUS_PATH = PKG_DATA_PATH.joinpath("corpus_test")
    features = [{"name": "tfidf"}]
    n = 4
    pipeline = Pipeline(CORPUS_PATH, "motif", features)

    pipeline.execute(
        method=args.method, n=args.ngram, plot=False
    )  # args.no_plot)
    print("here")
