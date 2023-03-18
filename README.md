# jigsaw-ml
Interoperable ML modules you can mix-and-match.

The core class of this is `Piece`.
A `LossFunction` or a `Module` are instances of `Piece`, and they can be recursively nested into `Composite` which are also an instance of `Piece`.

When a `Composite` runs, it will run all its internal components in topological order, which enables you to build increasingly complex computation flows without compromising readability.

## Testing
This repository uses pytest. You can run all tests with the following command from the main directory:
```bash
python -m pytest
```
