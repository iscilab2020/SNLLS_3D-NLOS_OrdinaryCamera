from SNLLS.model.world_model import *
from SNLLS.utils.utils import save_cs_plot, environment_fig
from SNLLS.optimizers.optmizers import GradientDescentJointREcovery, LiftingRecovery
from PIL import Image
import random
import argparse
from scipy.io import loadmat
import os
import imageio.v2 as imageio



seed = 32
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--M', type=int, default=128)
    parser.add_argument('--N', type=int, default=32)
    parser.add_argument("--num_points", type=int, default=10)
    parser.add_argument("--split_learning", type=int, default=1)
    parser.add_argument("--path_to_measurment", type=str, default="./data/real.mat")
    parser.add_argument("--max_iteration", type=int, default=100000)
    parser.add_argument("--path_to_save", type=str, default="./results/Real_Recons.jpg")
    
    
    args = parser.parse_args()
    
    M=(args.M, args.M); N=(args.N, args.N)
    
    

    device = "cuda" if torch.cuda.is_available() else "cpu"

    B = Pinspeck(camX_len=[0.808, 1.747], camZ_len=[0.05, 0.05+0.939], camDepth= 1.076, sceneDepth=0, scenePixels=N, 
                        camPixels=M, sceneX_len=[0, .708], sceneZ_len=[0.03, 0.436], occluders=None,
                        multiprocess=0, point_window=2, device=device, precision=32, cube = True, num_points=[args.num_points, 3, args.num_points])
    


    m = torch.Tensor(loadmat(args.path_to_measurment)["measurement"])
    
    
    N_occluder = len(B.points)
    Abar = B.GetABarModel(list(range(N_occluder)), return_vis=True, pinhole=False, return_pin=1)[0].reshape(-1, N_occluder, (N[0]*N[1]))
    print("Completed Forward Model")


    final_scene, final_occluder, loss_history, final_lambda_reg, final_bias = GradientDescentJointREcovery(B, m, 
                                                                                Abar=Abar, n_iterations=args.max_iteration, split_learning=args.split_learning)
    
    


    ss = torch.sigmoid(final_occluder)>0.5
    ss = [ i for i, j in enumerate(ss) if j ]

    g = {"scene":final_scene, "occ":final_occluder}
    torch.save(g, "./results/real_reconstructed_data.pt")



    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(111, projection='3d')
    
    environment_fig(B, occ=ss, measurement=m, ax=ax, Title="Estimated Occluder and Scene", scene=final_scene, save_fig=args.path_to_save)
    
    filenames = []
    for angle in range(0, 360, 2):  # Adjust the step for a smoother or faster rotation
        filename = f'./results/frame_{angle}.png'
        if os.path.exists(filename):
            filenames.append(filename)
            continue

        ax.view_init(30, angle)
        filenames.append(filename)
        fig.savefig(filename)
        plt.close(fig)
            
    with imageio.get_writer("./results/real_results.gif", mode='I', fps=20) as writer:  # Adjust fps as needed
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)

    # Remove the individual frame files
    import os
    for filename in filenames:
        os.remove(filename)
    


if __name__=="__main__":
    main()
    
