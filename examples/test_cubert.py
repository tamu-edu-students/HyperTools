import cuvis

print("howdy")



"""
current error
need to check to make sure files are copied to correct place 

  File "/workspaces/HyperTools/examples/test_cubert.py", line 1, in <module>
    import cuvis
  File "/cuvis.sdk/Python/src/cuvis/__init__.py", line 21, in <module>
    from .AcquisitionContext import AcquisitionContext
  File "/cuvis.sdk/Python/src/cuvis/AcquisitionContext.py", line 1, in <module>
    from . import cuvis_il
  File "/cuvis.sdk/Python/src/cuvis/cuvis_il.py", line 13, in <module>
    from . import _cuvis_pyil
ImportError: cannot import name '_cuvis_pyil' from partially initialized module 'cuvis' (most likely due to a circular import) (/cuvis.sdk/Python/src/cuvis/__init__.py)


"""