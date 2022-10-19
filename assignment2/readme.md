Remark:

- Q1: Fully-connected Neural Network
  - `forward` and `backward` function (for each layer), which can automatically compute loss and gradient for deep neural nets.
  - Use `forward` and `backward` API to implement 2-layer neural network, general (arbitrary number of hidden layers) neural network
  - Various parameter updates: Momentum, RMSProp, Adam
- Q2: Batch Normalization
  - Implement BatchNorm `forward` and `backward` function. **Requires drawing computation graph and calculating gradients manually**
  - Use these API to add BatchNorm to FC
- Q3: Dropout
  - Implement dropout `forward` and `backward` function and add them to FC
