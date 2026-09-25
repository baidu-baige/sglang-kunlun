# Contributing to SGLang Kunlun

Welcome to SGLang Kunlun! We warmly welcome community contributions. Feel free
to report issues or submit pull requests through the project's GitHub
repository. Before getting started, we highly recommend reading the contributing
guidance below and the repository [README](README.md).

## Contributing

Please refer to the repository documentation for the current development
environment and runtime setup. The [README](README.md) describes how to install
the project and configure the Kunlun runtime. Before submitting a change, make
sure that you understand the relevant SGLang and Kunlun runtime dependencies.

## Issues

We use the project issue tracker to record bugs, feature requests, and other
public discussions.

### Search Existing Issues First

Before opening a new issue, please search existing issues to check whether a
similar bug report or feature request already exists. This helps avoid
duplicates and keeps discussions focused.

### Reporting New Issues

When opening a new issue, please provide as much information as possible, such
as:

- A clear and detailed problem description
- Relevant logs or error messages
- The SGLang Kunlun, SGLang, Kunlun runtime, and model versions involved
- Reproduction commands and configuration, where applicable
- Code snippets, screenshots, or videos, if applicable

The more context you provide, the easier it will be for maintainers to
diagnose and resolve the issue.

## Pull Requests

We strongly welcome pull requests to help improve SGLang Kunlun.

### Submitting Pull Requests

All pull requests will be reviewed by the maintainers. Automated checks and
tests may be run as part of the review process. Once the checks pass and the
review is approved, the pull request can be merged according to the project
workflow.

Before submitting a pull request, please make sure that:

1. You create a topic branch from the current development base branch.
2. You update relevant code comments or documentation if APIs or behavior are
   changed.
3. You add the appropriate copyright notice to the top of any new file you
   author. Do not remove or replace the existing copyright/license header of
   any third-party file, and do not add our header to third-party files. If a
   file is adapted from another project, keep the upstream copyright and add a
   note stating that the file was modified.
4. Your code passes the applicable linting and style checks. This project uses
   [pre-commit](https://pre-commit.com/); install the hooks with
   `pre-commit install` before committing.
5. Your commits carry a `Signed-off-by:` line (`git commit -s`) as required by
   the [Developer Certificate of Origin](DCO).
6. Your changes are fully tested, including relevant tests under `test/` when
   applicable.
7. You include the test environment, commands, and results in the pull request
   description when they are relevant to the change.
8. You submit the pull request against the target branch requested by the
   maintainers.

## Code of Conduct

All contributors are expected to follow the
[Code of Conduct](CODE_OF_CONDUCT.md).

## License

By contributing to SGLang Kunlun, you agree that your contributions will be
licensed under the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0).
