from SNLLS.model.world_model import *
from SNLLS.utils.utils import save_cs_plot, environment_fig
from SNLLS.optimizers.optmizers import GradientDescentJointREcovery, LiftingRecovery
from PIL import Image
import random
import argparse
import torch



seed = 32
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--M', type=int, default=128)
    parser.add_argument('--N', type=int, default=32)
    parser.add_argument("--num_points", type=int, default=10)
    parser.add_argument("--split_learning", type=int, default=0)
    parser.add_argument("--path_to_scene", type=str, default="./data/smile.jpg")
    parser.add_argument("--snr", type=int, default=30)
    parser.add_argument("--max_iteration", type=int, default=500000)
    parser.add_argument("--path_to_save", type=str, default="./results/ADM_Estimated_1.jpg")
    
    
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    M=(args.M, args.M); N=(args.N, args.N)

    B = Pinspeck(camX_len=[0.808, 1.747], camZ_len=[0.05, 0.05+0.939], camDepth= 1.076, sceneDepth=0, scenePixels=N, 
                            camPixels=M, sceneX_len=[0, .708], sceneZ_len=[0.03, 0.436], occluders=None,
                            multiprocess=0, point_window=1, device=device, precision=32, cube = True, num_points=(args.num_points, 3, args.num_points))



    # Open an image file
    img = Image.open(args.path_to_scene)
    img_resized = img.resize(N)

    # Convert the image to a NumPy array
    scene = np.array(img_resized)


    x = torch.Tensor(np.array([scene[:, :, :3]]))
    x=x/x.max()


    
    cs = np.array([ 12, 13, 14, 15, 16, 17, 18, 19, 25, 35, 45, 56, 57, 58, 59, 55, 54,53,  52, 51 ])


    mode = B.GetPinspeckModel(cs, pinhole=False, return_vis=0)


    m = awgn(B.GetMeasurement(x, Model=mode), args.snr).cpu()

    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(121, projection='3d')


    environment_fig(B, occ=cs, measurement=m, ax=ax, scene=x,Title="True Occluder and Scene",  save_fig=False) # save_fig="True.jpg")


    N_occluder = len(B.points)
    Abar = B.GetABarModel(list(range(N_occluder)), return_vis=True, pinhole=False, return_pin=1)[0].reshape(-1, N_occluder, (N[0]*N[1]))


    final_scene, final_occluder, loss_history, final_lambda_reg, final_bias = GradientDescentJointREcovery(B, m, 
                                                                                Abar=Abar, n_iterations=args.max_iteration, split_learning=args.split_learning)


    ss = torch.sigmoid(final_occluder)>0.5
    ss = [ i for i, j in enumerate(ss) if j ]

    ax = fig.add_subplot(122, projection='3d')
    environment_fig(B, occ=ss, measurement=m, ax=ax, Title="Estimated Occluder and Scene", scene=final_scene, save_fig=args.path_to_save)



if __name__=="__main__":
    main()

    
   

    