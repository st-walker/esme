# Installation

## On the EuXFEL Control Room Machines

### Introduction

Distribution to machines on the EuXFEL control room machines is non-trivial for a few reasons.:

1. Conda is used to manage the Python environements, and there are no
   alternatives (no system Python, no python.org installer Python).
2. The conda environments are not shared across the various machines,
   so it is insufficient to make an
3. The general awfulness of Conda.


The goal is to escape Conda-world and bootstrap a completely
independent virtual environment on the shared file system so that the
GUI can be run from any machine in the control room and will not be affected 

The solution instead is to use Poetry to define all the dependencies
and to create a virtual environment in the shared section of the file
system so that it can hopefully be used on any machine.

### Instructions

```
conda create "--name=bootstrapping" "python=3.12"
conda install poetry
cd /Users/xfeloper/stwalker/esme
poetry build
```

### Updating a dependency

If you have a dependency



## Stable release

To install EuXFEL Slice Energy Spread Measurement Tool, run this command in your
terminal:

``` console
$ pip install esme-xfel
```

This is the preferred method to install EuXFEL Slice Energy Spread Measurement Tool, as it will always install the most recent stable release.

If you don't have [pip][] installed, this [Python installation guide][]
can guide you through the process.

## From source

The source for EuXFEL Slice Energy Spread Measurement Tool can be downloaded from
the [Github repo][].

You can either clone the public repository:

``` console
$ git clone git://github.com/st-walker/esme-xfel
```

Or download the [tarball][]:

``` console
$ curl -OJL https://github.com/st-walker/esme-xfel/tarball/master
```

Once you have a copy of the source, you can install it with:

``` console
$ pip install .
```

  [pip]: https://pip.pypa.io
  [Python installation guide]: http://docs.python-guide.org/en/latest/starting/installation/
  [Github repo]: https://github.com/%7B%7B%20cookiecutter.github_username%20%7D%7D/%7B%7B%20cookiecutter.project_slug%20%7D%7D
  [tarball]: https://github.com/%7B%7B%20cookiecutter.github_username%20%7D%7D/%7B%7B%20cookiecutter.project_slug%20%7D%7D/tarball/master


