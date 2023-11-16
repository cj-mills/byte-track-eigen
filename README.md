# ByteTrack-Eigen


## Introduction

ByteTrack-Eigen is a C++ implementation of the [ByteTrack](https://arxiv.org/abs/2110.06864) object tracking method, leveraging the Eigen library for high-performance matrix and vector operations. This library is designed for tracking objects in video frames using Kalman Filters and the Hungarian algorithm.

## Demo Video

https://github.com/cj-mills/byte-track-eigen/assets/9126128/8bcc137a-30e9-41a9-8e46-8ac0fba347bc

> Source Video by RDNE Stock project from Pexels: ([link](https://www.pexels.com/video/a-woman-giving-a-thumbs-up-10373924/))


## Features

- Object tracking using Kalman Filters and Hungarian algorithm.
- Handling of occlusions and re-identifying lost objects.
- Utilization of the Eigen library for optimized matrix and vector operations.



## Components

- `BaseTrack`: Foundational class for object tracking.
- `BoundingBoxTrackUtils`: Utility functions for track operations.
- `BYTETracker`: Main class for the BYTE tracking algorithm.
- `HungarianAlgorithmEigen`: Hungarian Algorithm implementation with Eigen.
- `KalmanBBoxTrack`: Kalman filter-based bounding box tracking.
- `KalmanFilter`: Kalman filter implementation for tracking bounding boxes in image space.
- `LinearAssignment`: Solves linear assignment problems using the Hungarian Algorithm.
- `BoundingBoxIoUMatching`: Utility functions for bounding box operations.

## Requirements

- CMake 3.20 or higher.
- C++17 or higher.
- [Eigen library](http://eigen.tuxfamily.org) (automatically downloaded by CMake).

## Building the Library

1. Clone the repository:
   ```bash
   git clone https://github.com/cj-mills/byte-track-eigen.git
   ```
2. Navigate to the project directory:
   ```bash
   cd byte-track-eigen
   ```
3. Build the project using CMake:
   ```bash
   mkdir build && cd build
   ```
   ```bash
   cmake ..
   ```



## Usage

Here is an example of how to use ByteTrack-Eigen in your project:

```cpp
// Example usage of ByteTrack-Eigen
#include "BYTETracker.h"

int main() {
    // Your code to initialize and use the tracker
}
```





## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgments

Special thanks to the Eigen library [team](https://gitlab.com/libeigen/eigen/-/project_members) and [authors](https://arxiv.org/abs/2110.06864) of the ByteTrack algorithm.

