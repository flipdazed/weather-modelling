Test Files
---

These files test each of the constituent models with the simple `MNIST` handwriting repository.

Tests are based on examples at [`DeepLearning.net`](http://deeplearning.net/tutorial/) and results can be directly compared to demonstrate the code is functioning as expected.

# Running Tests
Run tests from the parent (root) directory as described in `../README.md`

## Avoiding System Sleeping
All test scripts contain [`from sys import platform
if platform == "linux" or platform == "linux2":
    # linux
    pass
elif platform == "darwin":
    # OS X
    import caffeine
elif platform == "win32":
    # Windows...
    pass`](https://pypi.python.org/pypi/caffeine/0.2) which prevents the system from going into sleep during runtime.