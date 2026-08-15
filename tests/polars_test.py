import polars as pl


def main() -> None:
    df = pl.DataFrame(
        {
            "name": ["Alice", "Bob", "Charlie"],
            "score": [91, 78, 95],
        }
    )

    print(df)
    print(df.filter(pl.col("score") >= 90))


if __name__ == "__main__":
    main()