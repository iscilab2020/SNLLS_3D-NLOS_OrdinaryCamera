from SNLLS.model.world_model import *
from SNLLS.optimizers.optmizers import huristic_localization
import scipy
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--M', type=int, default=128)
parser.add_argument('--N', type=int, default=32)
parser.add_argument("--max_range", type=int, default=5)
parser.add_argument("--path_to_measurment", type=str, default="./data/real.mat")
parser.add_argument("--path_to_save", type=str, default="./results/grid_search_results.jpg")


args = parser.parse_args()
m = torch.Tensor(scipy.io.loadmat("real.mat")["measurement"])
real_mesh = trimesh.load("True_occluder_real_mesh.stl")

device = "cuda" if torch.cuda.is_available() else "cpu"

B = Pinspeck(camX_len=[0.808, 1.747], camZ_len=[0.05, 0.05+0.939], camDepth= 1.076, sceneDepth=0, scenePixels=N, 
                        camPixels=M, sceneX_len=[0, .708], sceneZ_len=[0.03, 0.436], occluders=None,
                        multiprocess=0, point_window=2, device=device, precision=32, cube = True, num_points=[3, 3, 3])



recons = huristic_localization(B, m, mesh=real_mesh, num_line=args.max_range)

plt.imshow(recons[0]/recons.max())
plt.savefig(args.path_to_save)





