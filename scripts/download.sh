#!/bin/bash
echo "Create downloads directory."
mkdir downloads

echo "Download files."
curl -o downloads/train-images-idx3-ubyte.gz http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
curl -o downloads/train-labels-idx1-ubyte.gz http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
curl -o downloads/t10k-images-idx3-ubyte.gz http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
curl -o downloads/t10k-labels-idx1-ubyte.gz http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz

echo "Extract gzip files."
gunzip -c downloads/train-images-idx3-ubyte.gz > downloads/train-images-idx3-ubyte
gunzip -c downloads/train-labels-idx1-ubyte.gz > downloads/train-labels-idx1-ubyte
gunzip -c downloads/t10k-images-idx3-ubyte.gz > downloads/t10k-images-idx3-ubyte
gunzip -c downloads/t10k-labels-idx1-ubyte.gz > downloads/t10k-labels-idx1-ubyte
