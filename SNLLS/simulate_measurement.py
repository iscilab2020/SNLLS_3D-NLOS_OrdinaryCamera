from SNLLS.model.world_model import *
from SNLLS.utils.utils import save_cs_plot, environment_fig
from SNLLS.optimizers.optmizers import GradientDescentJointREcovery, LiftingRecovery
from scipy.io import loadmat
import matplotlib.animation as animation
import imageio.v2 as imageio
import os


M=(128, 128); N=(32, 32)



device = "cuda" if torch.cuda.is_available() else "cpu"

B = Pinspeck(camX_len=[0.808, 1.747], camZ_len=[0.05, 0.05+0.939], camDepth= 1.076, sceneDepth=0, scenePixels=N, 
                    camPixels=M, sceneX_len=[0, .708], sceneZ_len=[0.03, 0.436], occluders=None,
                    multiprocess=0, point_window=2, device=device, precision=32, cube = True, num_points=[10, 3,10])

r = torch.load("/Users/fadlullahraji/Desktop/SNLLS_3D-NLOS_OrdinaryCamera/SNLLS/results/real_reconstructed_data.pt")

ss = torch.sigmoid(r["occ"])>0.5
ss = [ i for i, j in enumerate(ss) if j ]

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')

m = torch.Tensor(loadmat("./data/real.mat")["measurement"])


ax = environment_fig(B, occ=ss, measurement=m, ax=ax, Title="Estimated Occluder and Scene", scene=r["scene"], save_fig=None)

filenames = []
for angle in range(0, 360, 2):  # Adjust the step for a smoother or faster rotation
    # fig = plt.figure(figsize=(15, 15))
    # ax = fig.add_subplot(111, projection='3d')
    # environment_fig(B, occ=ss, measurement=m, ax=ax, Title="Estimated Occluder and Scene", scene=r["scene"], save_fig=None)
    filename = f'./results/frame_{angle}.png'
    if os.path.exists(filename):
        filenames.append(filename)
        continue

    ax.view_init(30, angle)
    filenames.append(filename)
    fig.savefig(filename)
    plt.close(fig)

with imageio.get_writer("./results/real_results.gif", mode='I', fps=20, loop=0) as writer:  # Adjust fps as needed
    for filename in filenames:
        image = imageio.imread(filename)
        writer.append_data(image)

# Remove the individual frame files if desired
import os
for filename in filenames:
    os.remove(filename)