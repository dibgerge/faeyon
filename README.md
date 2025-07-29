# faeyon


# TODO

- Update containers to return copies of themself to prevent weird syntax like

```python
var @ X[0]

1 >> var # error
```

This works ok for list/dict, but may need to change for var, and wrap the value in some container
class so value is passed by refernece between calls to the fae_var