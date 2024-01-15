# TOWARDS 3D COMPUTATIONAL PERSICOPY WITH AN ORDINARY CAMERA: A SEPARABLE NON-LINEAR LEAST SQUARES FORMULATION.
## 3D NLOS imaging with an ordinary camera by exploiting unknown hidden 3D occluders

<table>
  <!-- Titles -->
  <tr>
    <td><b>Real Experimental SetUp</b></td>
    <td><b>Reconstructed 3D Occluder and 2D scene</b></td>
  </tr>
  <!-- Content -->
  <tr>
    <!-- Image -->
    <td>
      <img src="SNLLS/data/real_setup.jpg" alt="Your Image Description" width="500"/>
    </td>
    <!-- GIF -->
    <td>
      <img src="SNLLS/results/real_results.gif" alt="Your GIF Description" width="500"/>
    </td>
  </tr>
</table>



## Citing SNLLS_3D-NLOS
If you use our codes or paper in your research, please cite this
```
@InProceedings{Fadlullah_2024_ICASSP,
author = {Fadlullah, Raji and John, Murray-Bruce},
}
```
## Getting Started

To run our results, follow these steps:

### Cloning the Repository

First, you need to clone the repository to your local machine. Open your terminal and run the following command:

```bash
git clone git@github.com:iscilab2020/SNLLS_3D-NLOS_OrdinaryCamera.git
```
Navigate to the cloned directory:

```bash
cd SNLLS_3D-NLOS_OrdinaryCamera
```

Create a Conda environment for the project:

```bash
conda create -n SNLLS python=3.8  # Replace with the required Python version
conda activate SNLLS
```
Install the software and the required dependencies:
```bash
pip install -e . 
```
Navigate into the Software:
```bash
cd SNLLS
```
Simulate and Reconstruct using the Lifting Method:
```bash
python3 test_lifing_method.py
```
Simulate and Reconstruct using the Alternating Minimization Method I:
```bash
python3 test_ADM_1.py
```
Simulate and Reconstruct using the Alternating Minimization Method II:
```bash
python3 test_ADM_2.py
```
Reconstruct Real World NLOS from captured 2D using the Alternating Minimization Method II:
```bash
python3 test_real_world.py
```
Use Saunder's et al.'s reconstruction method to search location and reconstruct the scene using a known shape of the occluder
```bash
python3 grid_search.py
```





