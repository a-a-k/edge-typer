# Makefile snippet for replaying the experiment matrix.
# Run `make matrix` to execute the full rate × replica grid (5×5) and aggregate results.

.PHONY: matrix aggregate clean

matrix:
	gh workflow run reproduce-matrix.yml && gh run watch

aggregate:
	python scripts/aggregate_replicas.py --replicas-dir runs --outdir site

clean:
	rm -rf runs site

