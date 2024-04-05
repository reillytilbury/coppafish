## Cannot open napari issues

If napari fails to open and you see an error such as 

```python
WARNING: composeAndFlush: makeCurrent() failed
```

in the terminal, here are a few suggestions that might fix the issue:

* In the conda environment, run `conda install -c conda-forge libstdcxx-ng`
* In the conda environment, run `conda install -c conda-forge libffi`.


## Pipeline crash

If the coppafish pipeline is crashing, first read the error message. If there is a suggestion about how to fix the 
issue in the config, try changing the config variable and run the pipeline again. If the suggestion does not make sense 
to you, feel free to reach out to the developers for help. If the error message is not clear to you, please 
[create an issue](https://github.com/reillytilbury/coppafish/issues/new) on GitHub!
