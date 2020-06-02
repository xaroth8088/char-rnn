# char-rnn
A simple Tensorflow 2.0-based character RNN, written by a guy who has no idea what he's doing

Based largely off the Tensorflow [text generation tutorial](https://www.tensorflow.org/tutorials/text/text_generation) and [Sherjil Ozair's Tensorflow 1 text generator](https://github.com/sherjilozair/char-rnn-tensorflow)

# Running this project
If you want GPU support, install [CUDA 10.1](https://developer.nvidia.com/cuda-10.1-download-archive-update2) and [cuDNN v7.6.5 for CUDA 10.1](https://developer.nvidia.com/rdp/cudnn-download).

Then, in your Python virtual environment:

`pip install -r requirements.txt`

## Docker
Alternately, a `Dockerfile` is included that will install the appropriate requirements into the image.

CUDA support is NOT included in this image, so running from inside the container will be very slow.  Also, it will throw a bunch of ignorable warnings due to CUDA not being present.

# Training the model
First, put a file named `input.txt` inside a directory somewhere.  The training script will create other files will be created alongside this input.

Then:

`python train.py --help` to see all available options.

For example:
`python train.py --data_dir your/path/here`

_Do not include `input.txt` in your `data_dir` value._

# Sampling
`python sample.py --help` to see all available options.

For example:
`python sample.py --data_dir your/path/here`

# Contributing
Pull requests are welcome.  Please create an issue before doing anything major just to check that it's something that would make sense for this project before putting in the bigger effort of coding it up.
