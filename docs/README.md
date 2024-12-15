# Documentation generation and test

## Local
1. Install the packages (only the first time)

```bash
# make sure in the docs folder
pip install -r requirements.txt
```

2. Generate documentation

```
make html
```

This may require pandoc which is installed on `woody` and you can install it on
your local machine if you prefer.

## Readthedoc

Once you tested on your local machine, you can push it to github. 
The readthedoc documentation will be automatically updated after push.

> ## Note
> 
> The readthedoc uses the package version (SNPmanifold) from PyPI not github,
> so you need to release on PyPI first if you want to update `autoclass` or 
> `automodule` in API, etc.
