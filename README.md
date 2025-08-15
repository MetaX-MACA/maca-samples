# MACA Samples
This repository contains a collection of reference implementations of fundamental GPU programming patterns with MXMACA.

## Getting Started

### Prerequisites

### Getting the MACA Samples

Using git clone the repository of MACA Samples using the command below.
```
git clone .....
```

Without using git the easiest way to use these samples is to download the zip file containing the current version by clicking the "Download ZIP" button on the repo page. You can then unzip the entire archive and use the samples.

### Building MACA Samples for Linux

The Linux samples are built using makefiles or CMakeLists. To use the makefiles, change the current directory to the sample directory you wish to build, and run make:
```
$ cd <sample_dir>
$ make
```
To use the CMakeLists, change the current directory to the sample directory you wish to build, and run:
```
$ cd <sample_dir>
$ mkdir build && cd build 
$ cmake .. && make  
```

## Samples list

### [0. Introduction](0_Introduction/README.md)
Getting started samples .If you are new to MACA, these are the best SDK samples to begin with. 


## Contributors Guide

We welcome your input on issues and suggestions for samples. This project collaboration adopts the standard Pull Request (PR) process mechanism. The PR process is the core mechanism for open-source project collaboration and code management. Below are the detailed steps for creating and handling PRs:

1   Fork the course repository
Copy the original repository to your Gitee account so that you can make modifications.
Click the "Fork" button in the top right corner of the page to copy this repository to your Gitee account.

2   Clone the course repository
On your local machine, use the git clone command to clone the forked repository of the course to your local environment:
```
mkdir <your-workspace>  && cd <your-workspace>
git clone ......
```

3   Create a new branch for development
Enter the directory of the cloned repository,Create a new branch (e.g., dev branch) for your creation or revision:
```
cd <your-workspace>
git checkout -b dev
```

4   Make modifications
Create or revise courses on your dev branch.After making modifications, add and commit your changes.
```
git add .
git commit -m "Describe your changes"
```

5   Push for modification
Push your changes to your forked repository
```
git push origin dev
```

6   Create a Pull Request
Submit your changes to the original repository maintainer and request a merge.
```
• Go back to your Gitee/Github account and find the repository you forked.
• Click the "New Pull Request" button.
• Select the branch you pushed and compare it with the main branch of the original repository.
• Fill in the title and description of the PR, explaining the content of your modifications and the purpose.
• Click the "Create Pull Request" button to submit the PR.
```

7   Process feedback
The project maintenance team or maintainer will review your PR and may raise suggestions for changes or questions.
Make necessary modifications based on the feedback, and repeat steps 4 to 6 until the PR is accepted.

8   PR has been merged
The project maintainer will merge your PR. Your changes are now part of the original repository.

9   Synchronize the upstream repository

To keep your Fork repository in sync with the upstream repository, you need to pull updates from the upstream repository regularly(recommended you check before git add).
```
Log in to your Gitee/Github account and find the repository you Forked.
Click the Sync fork button to synchronize the latest upstream Fork repository.
```

## Frequently Asked Questions

Answers to frequently asked questions about MACA can be found at [MXMACA Programming Forum](https://developer.metax-tech.com/forum/c/bian-cheng/14/).

## License

License. See [LICENSE](LICENSE)