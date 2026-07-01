# Contributing

This is a research repository. Maintenance bandwidth is limited, but bug reports
and questions are welcome.

## Bug Reports

Open a GitHub Issue with:
- A minimal reproduction (script + command that triggers the bug)
- Your environment: OS, CUDA version, PyTorch version (`python -c "import torch; print(torch.__version__)"`)
- The full traceback

## Questions

Open a GitHub Discussion.

## Code Style

- Formatter/linter: `black`, `flake8`, `isort` (see `setup.py` dev extras)
- Line length: 100 characters
- Run before submitting a PR:

```bash
black --line-length 100 part1_generation/ part2_navigation/ part2_manipulation/ simulator/
flake8 part1_generation/ part2_navigation/ part2_manipulation/ simulator/ --max-line-length=100
isort --line-length 100 part1_generation/ part2_navigation/ part2_manipulation/ simulator/
```

## Running Tests

There is no unit test suite in this repository yet. Integration testing is done
by running the inference and simulator scripts against real checkpoints — see
`scripts/quick_start.sh` and the per-part README files for the canonical
validation commands.

## Pull Requests

This is primarily a paper reproduction repository. PRs are welcome for:
- Bug fixes in scripts or configs
- Documentation improvements
- Additional evaluation scripts

PRs that restructure the training pipeline or change reported results will not
be accepted, as they would invalidate the paper's claims.

Please follow [Conventional Commits](https://www.conventionalcommits.org/) for
commit messages: `type(scope): description` where type is one of `feat`, `fix`,
`docs`, `refactor`, `chore`, `test`.

## Scope of Maintenance

This repository is released for reproducibility. The author is a PhD student
at Skoltech; response times may be slow. Critical reproducibility bugs (wrong
checkpoint, broken eval pipeline) will be prioritized.
