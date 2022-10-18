#!/usr/bin/env python3

import urllib.request
import tarfile


url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
fileobj = urllib.request.urlopen(url)

print("Downloading...")
with tarfile.open(fileobj=fileobj, mode="r|gz") as tar:
    tar.extractall()
