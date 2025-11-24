# AI and Cybersecurity - Demo DarkVec

In this repository, we provide a demo version of DarkVec, a Word2Vec-base model trained to recognize coordinated attackers in a Darknet.

## Overview

The repository contains the following key components:

- **Jupyter Notebook**: This notebook demonstrates how to use the DarkVec model for identifying coordinated attackers.
- **src folder**: Containing the source code to load and utilize the DarkVec model.
- **data**: 30 days of darknet logs used for training the model.

**Notice**: The DarkVec model in the demo uses i-DarkVec, an improvement over the original DarkVec model.
Specifically, i-DarkVec uses an incremental training approach, allowing it to adapt to new data without retraining from scratch. Old data will be used as anchors to i) maintain knowledge from previous training and ii) stabilize the training process.

## Getting Started

To get started with the demo, follow these steps:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/MatteoBoffa/darkvec_demo.git
   ```
2. **Install dependencies**:
   Create a virtual environment
   ```bash
   python -m venv venv
   ```
3. **Activate the virtual environment**:
   - On Windows:
     ```bash
     venv\Scripts\activate
     ```
   - On macOS/Linux:
     ```bash
     source venv/bin/activate
     ```
4. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
5. **Run the Jupyter Notebook**:
   Open the Jupyter Notebook located in the root directory and run the cells to see how the DarkVec model works.

## References

If you use this demo or the DarkVec model in your research, please cite the following paper:

```
@article{gioacchini2023darkvec,
  title={i-darkvec: Incremental embeddings for darknet traffic analysis},
  author={Gioacchini, Luca and Vassio, Luca and Mellia, Marco and Drago, Idilio and Houidi, Zied Ben and Rossi, Dario},
  journal={ACM Transactions on Internet Technology},
  volume={23},
  number={3},
  pages={1--28},
  year={2023},
  publisher={ACM New York, NY}
}
```
