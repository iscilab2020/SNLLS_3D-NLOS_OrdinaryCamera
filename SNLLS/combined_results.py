import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np


def main():


    """
        Uncomment to Run the Reconstruuction Reults in Parallel
        
    """
    # import subprocess

    # # Define the command to activate the Conda environment and then run the script
    # command1 = "python3 test_AGD.py --max_iteration=50000"
    # command2 = "python3 test_AGD.py --split_learning=0 --path_to_save=ADG.jpg --max_iteration=50000"
    # command3 = "python3 test_kronecker.py --max_iteration=5000"
    # command4 = "python3 test_real_world.py --max_iteration=50000"

    # # Start all processes with different CUDA devices
    # processes = []
    # processes.append(subprocess.Popen(command1, shell=True, env={"CUDA_VISIBLE_DEVICES": "0"}))
    # processes.append(subprocess.Popen(command2, shell=True, env={"CUDA_VISIBLE_DEVICES": "1"}))
    # processes.append(subprocess.Popen(command3, shell=True, env={"CUDA_VISIBLE_DEVICES": "2"}))
    # processes.append(subprocess.Popen(command4, shell=True, env={"CUDA_VISIBLE_DEVICES": "3"}))

    # # Wait for all processes to complete
    # for p in processes:
    #     p.wait()

    saved_fig = ["./results/Estimated_Kronecker.jpg", "./results/AMA_Estimated_1.jpg",  "./results/AMA_Estimated_2.jpg",  "./results/Real_Recons.jpg",]
    names = ["Lifting Method", "Alternating Minimization Method", "Alternating Minimization Method with Split, b", "Real Reconstruction Results"]

    plt.figure(figsize=(10, len(saved_fig) * 5))


    for i, fig in enumerate(saved_fig):

        img = np.asarray(Image.open(fig))

        plt.subplot(len(saved_fig), 1, i + 1)
        plt.imshow(img[400:1000, 200:])
        plt.axis('off')  # Hide axis
        plt.title(f"{names[i]}")

    plt.tight_layout()
    plt.savefig("combined_results")



if __name__ == "__main__":
    main()