from SNLLS.model.world_model import *
import matplotlib.pyplot as plt
# import cv2

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    


def environment_fig(B, occ=None, measurement=None, scene=None, ax =None, show=False, save_fig=False, Title=None):
        
    
    if isinstance(measurement, torch.Tensor):
        measurement = measurement.cpu().numpy()


    if isinstance(scene, torch.Tensor):
        scene = scene.cpu().numpy()

    if ax is None:
        fig = plt.figure(figsize=(15, 15))
        ax = fig.add_subplot(111, projection='3d')

    if occ is not None:  
        boxes = B.points
        cs = occ
        occluders = []
        for b in cs:
            b = B.ids[b]
            point = boxes[b]
            box = point[0], point[1], point[2], point[0]+point[3], point[1]+point[4], point[2]+point[5]
            tupleList, vertices = B.get_verts(box)
            vertices = vertices.tolist()
            occluders.append(box)
            
        

            poly3d = [[tupleList[vertices[ix][iy]] for iy in range(len(vertices[0]))] for ix in range(len(vertices))]
        
            ax.add_collection3d(Poly3DCollection(poly3d,color="black", linewidths=4, alpha=0.9))

    if scene is not None:
        xx=np.flipud(scene[0]/scene.max())
        xs = np.linspace(B.sceneX_len[0], B.sceneX_len[1], xx.shape[0])

        ys = np.zeros_like(xs)
        xs, ys = np.meshgrid(xs, ys)

        zs = np.linspace(B.sceneZ_len[0], B.sceneZ_len[1], xx.shape[1])
        zs = np.tile(zs, (xx.shape[0], 1)).T

        ax.plot_surface(xs, ys, zs, facecolors=xx, rstride=1, cstride=1)

    if measurement is not None:
        mm=measurement[0]/measurement.max()
        xs = np.linspace(B.camX_len[0], B.camX_len[1], mm.shape[0])

        ys = np.ones_like(xs)*B.camDepth
        xs, ys = np.meshgrid(xs, ys)

        zs = np.linspace(B.camZ_len[0], B.camZ_len[1], mm.shape[1])
        zs = np.tile(zs, (mm.shape[0], 1)).T

        # ax.scatter(xs, ys, zs)
        ax.plot_surface(xs, ys, zs, facecolors=mm, rstride=1, cstride=1)



    x_wall = np.ones((50, 50))*1.1
    y_wall = np.linspace(0, 0.59, 50)
    z_wall = np.linspace(0, 0.6, 50)
    y_wall, z_wall = np.meshgrid(y_wall, z_wall)

    # Plot the wall
    ax.plot_surface(x_wall, y_wall, z_wall, color='white', alpha=.9)


    ax.set_xlabel('X')
    ax.set_ylabel('Y - Depth')
    ax.set_zlabel('Z -Height')

    if Title is not None:
        ax.set_title(str(Title))



    ax.view_init(30, 120)

    if show:
        plt.show()

    if save_fig:
        plt.savefig(save_fig)


    return fig if ax is None else ax

        


def save_cs_plot(B, cs, fig_name=False):
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    
    fig = plt.figure(figsize=(21, 21))
    ax = fig.add_subplot(111, projection='3d')


        
    boxes = B.points

    occluders = []
    for b in cs:
        b = B.ids[b]
        point = boxes[b]
        box = point[0], point[1], point[2], point[0]+point[3], point[1]+point[4], point[2]+point[5]
        tupleList, vertices = B.get_verts(box)
        vertices = vertices.tolist()
        occluders.append(box)
        
    

        # tupleList = list(zip(x, y, z))
        # print(tupleList)
        # print(faces)

        poly3d = [[tupleList[vertices[ix][iy]] for iy in range(len(vertices[0]))] for ix in range(len(vertices))]
    
        ax.add_collection3d(Poly3DCollection(poly3d, facecolors='g', linewidths=1, alpha=0.5))
        
    xs = B.scenePoints[:, 0, 0].cpu()
    ys = B.scenePoints[:, 0, 1].cpu()
    zs = B.scenePoints[:, 0, 2].cpu()
    ax.scatter(xs, ys, zs)

    xs = B.camPoints[:, 0].cpu()
    ys = B.camPoints[:, 1].cpu()
    zs = B.camPoints[:, 2].cpu()
    ax.scatter(xs, ys, zs)


    if not fig_name:
        return fig

    # plt.show()
    plt.savefig(fig_name)


