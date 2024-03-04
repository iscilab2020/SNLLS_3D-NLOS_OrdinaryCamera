from SNLLS.model.world_model import *
from SNLLS.utils.utils import save_cs_plot, environment_fig
from SNLLS.optimizers.optmizers import GradientDescentJointREcovery, LiftingRecovery
from utils import environment_fig
import random
from PIL import Image


seed = 2023
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)


def main():
    M=(128, 128); N=(10, 10)
    
    
    device = "cuda" if torch.cuda.is_available() else "cpu"

    B = Pinspeck(camX_len=[0.808, 1.747], camZ_len=[0.05, 0.05+0.939], camDepth= 1.076, sceneDepth=0, scenePixels=N, 
                            camPixels=M, sceneX_len=[0, .708], sceneZ_len=[0.03, 0.436], occluders=None,
                            multiprocess=0, point_window=1, device=device, precision=32, cube = True, num_points=(3, 3, 3))




    import matplotlib.pyplot as plt
    # import cv2

    x = torch.zeros(1, N[0], N[1], 3)
    x[:, 3:4, 3:5, :1] = 1.
    x[:, 5:7, 8:9, :2] = 1.
    x[:, 7:9, 0:5, :3] = 1.


    img = Image.open("./data/smile.jpg")
    img_resized = img.resize(N)

    # Convert the image to a NumPy array
    scene = np.array(img_resized)


    x = torch.Tensor(np.array([scene[:, :, :3]]))
    x=x/x.max()



    cs = np.array([ 14, 15,16, 17, 18, 19,20, 24,])# 31, 38, 42, 43, 44, 45, 46, 47, 48,  ])

    mode = B.GetPinspeckModel(cs, pinhole=False)




    # # print(B.points)

    m = awgn(B.GetMeasurement(x, Model=mode), 40).cpu()

    print("Started Lifting")

    xx, occ = LiftingRecovery(B, m, n_iteration=200)


    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(121, projection='3d')

    environment_fig(B, occ=cs, measurement=m, ax=ax, scene=x,Title="True Occluder and Scene",  save_fig=False) # save_fig="True.jpg")



    ax = fig.add_subplot(122, projection='3d')
    xx = torch.Tensor(xx)
    xx=torch.relu(xx[None]/xx.max())


    environment_fig(B, occ=occ, measurement=m, ax=ax, Title="Estimated Occluder and Scene", scene=xx, save_fig="./results/Estimated_Kronecker.jpg")


if __name__=="__main__":
    main()
    