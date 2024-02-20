import pandas as pd


class CanonData:
    def __init__(self, path):
        self.data = pd.read_csv(path, index_col=0)
        # Date segments
        periods = [
            [1800, 1826],
            [1827, 1850],
            [1851, 1869],
            [1870, 1899],
            [1900, 1945],
            [1946, 2024],
        ]
        self.periods = pd.DataFrame(periods, columns=["start", "end"])
        # Add period column
        self.data["period"] = self.data["date_publication"].apply(
            lambda x: self.get_seg_from_date(x)
        )
        self.data.doc_id = self.data.doc_id.str.replace(".csv", "")

    def get_seg_from_date(self, date):
        return self.periods[
            (self.periods["start"] <= date) & (date <= self.periods["end"])
        ].index[0]
