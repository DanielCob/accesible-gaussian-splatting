# Accessible Gaussian Splatting

An accessible implementation of 3D Gaussian Splatting designed for non-expert users, featuring a simplified, educational, and fully automated pipeline with a Google Colab interface.

## 🚀 Quick Start with Google Colab

**[Open in Google Colab](https://colab.research.google.com/github/DanielCob/accesible-gaussian-splatting/blob/main/accesible_gaussian_splatting.ipynb)**

The Colab notebook provides a complete, user-friendly interface for creating 3D reconstructions from video:

1. **Upload a video** - Simply record a short video orbiting around an object or scene
2. **Automated preprocessing** - The notebook automatically extracts frames and prepares the data
3. **Training** - Trains a 3D Gaussian Splatting model with optimized parameters
4. **Export results** - Download your 3D model for visualization in various tools

No complex setup, dependencies, or command-line experience required!

## About This Project

This repository adapts the [original 3D Gaussian Splatting implementation](https://github.com/graphdeco-inria/gaussian-splatting) by Kerbl et al. to make it more accessible for educational purposes and non-expert users.

**Developed by:** Daniel Cob Beirute, Instituto Tecnológico de Costa Rica (2025)

**Based on:** "3D Gaussian Splatting for Real-Time Radiance Field Rendering" by Kerbl, Kopanas, Leimkühler, and Drettakis

## Features

- **Google Colab Integration** - Run everything in your browser with free GPU access
- **Simplified Workflow** - Automated pipeline from video to 3D model
- **Educational Focus** - Clear documentation and user-friendly interface
- **No Local Setup Required** - All dependencies handled automatically in Colab

## Citation

If you use this work, please cite the original 3D Gaussian Splatting paper:

```bibtex
@Article{kerbl3Dgaussians,
  author       = {Kerbl, Bernhard and Kopanas, Georgios and Leimk{\"u}hler, Thomas and Drettakis, George},
  title        = {3D Gaussian Splatting for Real-Time Radiance Field Rendering},
  journal      = {ACM Transactions on Graphics},
  number       = {4},
  volume       = {42},
  month        = {July},
  year         = {2023},
  url          = {https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/}
}
```

## Additional Resources

- [Original 3D Gaussian Splatting Repository](https://github.com/graphdeco-inria/gaussian-splatting)
- [Project Webpage](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/)
- [Research Paper](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/3d_gaussian_splatting_high.pdf)
- [Viewer by Camenduru](https://colab.research.google.com/github/camenduru/gaussian-splatting-colab/blob/main/gaussian_splatting_viewer_colab.ipynb) - for visualizing exported models

## License

This project inherits the license from the original 3D Gaussian Splatting implementation. Please see [LICENSE.md](LICENSE.md) for details.
