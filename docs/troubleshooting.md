## Cannot open napari issues

If napari fails to open and you see an error such as 

```python
WARNING: composeAndFlush: makeCurrent() failed
```

in the terminal, here are a few suggestions that might fix the issue:

* In the conda environment, run `conda install -c conda-forge libstdcxx-ng`
* In the conda environment, run `conda install -c conda-forge libffi`.


## Pipeline crash

If the coppafish pipeline is crashing, we recommend sending us an issue on GitHub!
