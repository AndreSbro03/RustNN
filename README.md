# Logic Gates Neural Network AI 🔗🧠

A Neural Network written in **Rust** that can learn and solve **Logic Gates** like AND, OR, XOR, and NAND using **linear regression**. The project also includes a prototype to upscale images, although it is currently slow due to lack of optimization.

This project is a great demonstration of how neural networks can be applied to simple logic problems, with a future potential for image processing tasks.

### Features
- 🧠 **Neural Network for Logic Gates**: Solves standard logic gates (AND, OR, XOR, NAND).
- 🎛 **Adjustable Learning Rate**: Use a slider to experiment with different learning rates during training.
- 🖼 **Image Upscaling Prototype**: Neural network-powered image processing, though it's in an experimental and slow stage.

Built using **raylib** for window management. Check it out at [raylib.com](https://www.raylib.com/).

## Screenshots

![App Screenshot](screenshots/240921_12h35m01s_screenshot.png)


[Watch NAND ScreenRecord](https://github.com/user-attachments/assets/a078211d-7b59-4849-b4d1-4659f6aa77e6)


## Installation

To get started with the project, follow these steps:

1. Clone the repository.
2. Run the project using Cargo:

   ```bash
   cargo run --bin logicGates -v

## Globals and variables

Globals:

- `DATA` Desire Logic Gate to learn `AND`, `OR`, `XOR`, `NAND` ...

- `LEN_ARC` Number of hidden layers 

- `ARC` Format of the hidden layers 

- `NUM_NEURONS` Total number of neurons 

## Image Upscaling Prototype

The project also includes an experimental neural network-based image upscaling feature. Although it shares the same architecture as the logic gate network, the current implementation is unoptimized and runs slowly. 

This prototype can serve as a foundation for future developments in image processing. Optimizing this neural network could improve its performance and expand its use in more complex image tasks.

## Acknowledgements

This project was inspired by [Tsoding's Neural Network Project](https://youtu.be/PGSba51aRYU?si=QogqQE1VkJ9B_6Uy). Special thanks to Tsoding for the inspiration and guidance!

---

Feel free to contribute or experiment with the code to explore different possibilities in logic gate learning and neural networks.


