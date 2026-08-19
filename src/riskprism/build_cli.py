"""CLI entry point: riskprism-build."""

import argparse

from riskprism.model.build import build_model


def main() -> None:
    p = argparse.ArgumentParser(
        prog="riskprism-build",
        description="Build US equity factor risk model artifacts from public data.",
    )
    p.add_argument("--tickers", nargs="*", default=None,
                   help="Explicit ticker list (default: EDGAR universe)")
    p.add_argument("--max-names", type=int, default=None,
                   help="Cap universe size (EDGAR ordering ~ market cap)")
    p.add_argument("--provider", default="yahoo", choices=["yahoo", "stooq", "tiingo"])
    p.add_argument("--start", default=None, help="History start (default: 4y before end)")
    p.add_argument("--end", default=None, help="History end (default: today)")
    p.add_argument("--out", default="artifacts", help="Artifact output directory")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()

    build_model(
        tickers=args.tickers,
        provider=args.provider,
        start=args.start,
        end=args.end,
        max_names=args.max_names,
        artifacts_dir=args.out,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
