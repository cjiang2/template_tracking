# template_tracking
My Python Implementation of Hyperplane Template Tracking Algorithm, and other Template Matching based Tracking algorithm with a touch of Machine Learning methods.

## Introduction

This repo focuses on simple, straightforward implementations of some old Template Matching based Tracking algorithms. Those trackers are also good study cases for classic combination of Machine Learning and Computer Vision.

This repo is intended for learning only. For general and high performance purpose, feel free to check [List of Deep Learning based Tracking Methods](https://github.com/foolwood/benchmark_results).

For more complex and efficient use cases of registration-based tracking methods, please check [MTF](http://webdocs.cs.ualberta.ca/~vis/mtf/index.html).

Some key papers in regards of the implementations:

[Real Time Robust Template Matching](https://pdfs.semanticscholar.org/7fbc/4c4f01eb9716959ffef8b4a620a3d1c38577.pdf).

[Realtime Registration-Based Tracking via Approximate Nearest Neighbour Search](http://www.roboticsproceedings.org/rss09/p44.pdf)

## Requirements
- Python 3.6
- Numpy
- OpenCV
- scikit-learn
- pyflann, [modified version]((https://github.com/nashory/pyflann)) for python3 support

## Demo
Some demo results for Hyperplane Template Tracker:

<center>
![](images/result_book3.gif)
![](images/result_box.gif)
![](images/result_cereal.gif)
![](images/result_towel.gif)
</center>

## TO-DOs
[] Original Uniform Corner Sampling.
[] Original region sampling.
[] Probablistic image difference feature projection.
