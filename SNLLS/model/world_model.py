
import numpy as np
from numpy import ndarray
from joblib import Parallel, delayed
import torch
import matplotlib.pyplot as plt
from torch import einsum
from einops import rearrange
import open3d as o3d
from torch import nn
import trimesh
from joblib import Parallel, delayed





def line_intersection_point(line1, line2):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
       raise Exception('lines does not intersect')

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return x, y
    

def occ_range(D, y, xs, xm):
    x_min = (y[1]/D)*(xm[0] - xs[0]) + xs[0]
    x_max = (y[0]/D)*(xm[1] - xs[1]) + xs[1]
    return (x_min, x_max)



def find_intercepting_line(scene, cams, depth):

    D = depth
    x0, x1 = scene[0], cams[-1]
    y0, y1 = 0, D
    line1 = [[x0, y0], [x1, y1]]


    x0, x1 = scene[1], cams[0]
    y0, y1 = 0, D
    line2 = [[x0, y0], [x1, y1]]

    return line1, line2

def find_parallel_line(scene, cams, depth):

    D = depth
    x0, x1 = scene[0], cams[0]
    y0, y1 = 0, D
    line1 = [[x0, y0], [x1, y1]]


    x0, x1 = scene[1], cams[1]
    y0, y1 = 0, D
    line2 = [[x0, y0], [x1, y1]]

    return line1, line2



def awgn(s, SNR_min=100, SNR_max=None, L=1, return_snr=False):
    shape = s.shape
    """
    AWGN channel
    Add AWGN noise to input signal. The function adds AWGN noise vector to signal 's' to generate a resulting signal vector 'r' of specified SNR in dB. It also
    returns the noise vector 'n' that is added to the signal 's' and the power spectral density N0 of noise added
    Parameters:
        s : input/transmitted signal vector
        SNRdB : desired signal to noise ratio (expressed in dB) for the received signal
        L : oversampling factor (applicable for waveform simulation) default L = 1.
    Returns:
        r : received signal vector (r=s+n)
    """
    s = torch.Tensor(s)
    device = s.device

    assert len(shape) == 4, f"Expect 4 Dim, Got {shape} Dim to use"

    if SNR_max:
        SNRdB = torch.randint(low=SNR_min, high=SNR_max, size=(shape[0],)).to(device)
    else:
        if isinstance(SNR_min, int):
            return_snr = False

            SNRdB = torch.ones((shape[0],)).to(device)*SNR_min
        else:
            SNRdB = torch.tensor(SNR_min).to(device)

    s = torch.reshape(s, [s.shape[0], -1])
    gamma = 10**(SNRdB/10)


    P = L *torch.sum(torch.abs(s)**2, dim=1)/s.shape[-1]
    N0 = P/gamma
    n = torch.sqrt(N0/2).unsqueeze(1)*torch.rand(s.shape).to(device)
    s = s+n
    if return_snr:
        return torch.reshape(s, shape), SNRdB
    else:
        return torch.reshape(s, shape)


def normalize(s):
    shape = s.shape
    s = torch.reshape(s, (shape[0], -1))
    s = torch.divide(s, s.max(1)[0].unsqueeze(1))
    return torch.reshape(s, shape)



class Forward_Model(nn.Module):
    """
    Simulation Class for Non Line of Sight ForWard Model  
    
    Args:
        CamX_len ( list ) --> Distance of the Camera View along X (e.g [0, 6]: starts at 0m, and ends at 6m)
        CamZ_len ( list ) --> Distance of the Camera View along Z (e.g [0, 6]: starts at 0m, and ends at 6m)
        CamDepth (int ) --> The Depth of the Scene to the Visible wall in Meters 
        SceneDepth [Optional in 2D reconstruction](int) --> Depth from the Camera to the Scene
        scenePixels ( List ) --> Pixel Size for the Hidden Scene
        camPixels ( List ) --> Pixel Size for the Visible Wall
        sceneX_len ( list ) --> Distance of the Hidden View along X (e.g [0, 6]: starts at 0m, and ends at 6m)
        sceneZ_len ( list ) --> Distance of the Hidden View along Z (e.g [0, 6]: starts at 0m, and ends at 6m)
        Occluders  [Optional] [List[list]] --> Previously Known Occluder location (e.g [x, y, z, width, heigth])
        multiprocess (Int) --> Number of Jobs To Use Multiprocessor or Not
        point_window (Int) --> Pixel wise Discritization of the Hidden Scene
        device (str ("cpu", "cuda")) --> on what Device should the Simulation Run
        Precision (int)[16, or 32] --> Precision of Computational Resulte (32 for float 32, 16 for bfloat16)
        
    """

    def __init__ (self, 
                camX_len:list=[0.808, 1.747], camZ_len:list=[0.05, 0.729], camDepth:float= 1.076, sceneDepth:float=0, scenePixels:list=[10, 10], 
                        camPixels:list=[50, 50], sceneX_len:list=[0, .708], sceneZ_len:list=[0.03, 0.436], occluders:list=None,
                        multiprocess:int=0, point_window:int=8, device:str="cpu", precision:int=32) -> None:


        if precision==16:

            self.default_dtype = np.float16
            self.torch_dtype = torch.bfloat16


        else:
            self.default_dtype = np.float32
            self.torch_dtype = torch.float32


        self.multiprocess = multiprocess

        self.M1, self.M2 = camPixels[-1::-1]  # Using column Fist Matrix for all computation
        self.N1, self.N2 = scenePixels[-1::-1]

        self.sceneX_len = sceneX_len
        self.sceneZ_len = sceneZ_len


        self.camX_len = camX_len
        self.camZ_len = camZ_len

        self.camDepth = camDepth
        self.sceneDepth = sceneDepth
        self.device = device

        self.occluders = occluders
        self.point_window = point_window

        # Grab Depth Array for both Camera and hidden Scene
        self.camDepth_array = np.full(shape=(1, self.M1*self.M2 ), fill_value=camDepth, dtype=self.default_dtype)
        self.sceneDepth_array = np.full(shape=(1, self.N1*self.N2 ), fill_value=sceneDepth, dtype=self.default_dtype)

        # Discritize the Visible Wall Space
        self.cam_x = np.linspace(camX_len[0], camX_len[1], self.M1+1, dtype=self.default_dtype)[:-1]
        self.cam_z = np.linspace(camZ_len[0], camZ_len[1], self.M2+1, dtype=self.default_dtype)[:-1]
        self.cam_x, self.cam_z = self.cam_x+((self.cam_x[1] - self.cam_x[0])/2.),  self.cam_z+((self.cam_z[1] - self.cam_z[0])/2.)
        self.camPoints = self.cast(self.__measurement_points).to(self.device)

        # Discritize the Hidden Scene 
        self.scene_x = np.linspace(sceneX_len[0], sceneX_len[1], self.N1, dtype=self.default_dtype)
        self.scene_z = np.linspace(sceneZ_len[0], sceneZ_len[1], self.N2, dtype=self.default_dtype)
        
        if point_window==0:
            self.scene_x = np.linspace(sceneX_len[0], sceneX_len[1], self.N1+1, dtype=self.default_dtype)[:-1]
            self.scene_z = np.linspace(sceneZ_len[0], sceneZ_len[1], self.N2+1, dtype=self.default_dtype)[:-1]
            self.scene_x, self.scene_z = self.scene_x+((self.scene_x[1] - self.scene_x[0])/2.), self.scene_z+((self.scene_z[1] - self.scene_z[0])/2.)
            
        self.scenePoints = self.cast(self.__hidden_scene_points).to(self.device)

        self.inverse_model = None
        self.reg_init = None
        self.AtA = None
        self.Model = None


    def cast(self, inputs, type=None):
        """
        Helper Function for Casting Array into Defined Dtype
        """
        if type is None:
            type = self.torch_dtype

        if torch.is_tensor(inputs):
            inputs = inputs.type(type)
        else:
            inputs =  torch.as_tensor(inputs, dtype=type)

        if inputs.device != self.device:
            return inputs.to(self.device)

        return inputs

    
    def OccMinMax(self, plot_range=5, plot=False, ):
        """
        Function for Calculating the Depth of Occuler Space Using Line properties
        Args: To plot Oclcuder SPace using Matplot Lib
       
            
        Returns:
            y_min: The minimum value of the depth for occluders
            y_max: The minimum value of the depth for occluders
            
        """
        D = self.camDepth
        x_line1, x_line2 = find_intercepting_line(self.sceneX_len, self.camX_len, self.camDepth)
        z_line1, z_line2 = find_intercepting_line(self.sceneZ_len, self.camZ_len, self.camDepth)

        x_int, yx_int = line_intersection_point(x_line1, x_line2)
        z_int, yz_int = line_intersection_point(z_line1, z_line2)


        xp_line1, xp_line2 = find_parallel_line(self.sceneX_len, self.camX_len, self.camDepth)
        zp_line1, zp_line2 = find_parallel_line(self.sceneZ_len, self.camZ_len, self.camDepth)

        
        y_avg = max(yx_int, yz_int)



        y_min, y_max = y_avg, y_avg + ((D-y_avg)/2)
        x_min, x_max = occ_range(D, (y_min, y_max), self.sceneX_len, self.camX_len)
        z_min, z_max = occ_range(D, (y_min, y_max), self.sceneZ_len, self.camZ_len)



        if plot:

            x = np.linspace(x_min, x_max, plot_range, dtype=self.default_dtype)
            y = np.linspace(y_min, y_max, plot_range, dtype=self.default_dtype)
            z = np.linspace(z_min, z_max, plot_range, dtype=self.default_dtype)

            print((x_min, x_max), (y_min, y_max), (z_min, z_max))

            x, y, z = np.meshgrid(x, y, z)

            xyz = np.column_stack([x.flatten("F"), y.flatten("F"), z.flatten("F")])


            fig = plt.figure(figsize=(12, 12))  
            ax = fig.add_subplot(111, projection='3d')

            ax.plot( np.array(x_line1)[:, 0], np.array(x_line1)[:, 1], np.zeros_like(x_line1)[:, 1]); 
            ax.plot(np.array(x_line2)[:, 0], np.array(x_line2)[:, 1], np.zeros_like(x_line1)[:, 1])
            ax.plot(np.array(xp_line1)[:, 0], np.array(xp_line1)[:, 1], np.zeros_like(x_line1)[:, 1]); 
            ax.plot(np.array(xp_line2)[:, 0], np.array(xp_line2)[:, 1], np.zeros_like(x_line1)[:, 1])
            ax.plot(np.zeros_like(x_line1)[:, 1], np.array(z_line1)[:, 1], np.array(z_line1)[:, 0]); 
            ax.plot(np.zeros_like(x_line1)[:, 1], np.array(z_line2)[:, 1], np.array(z_line2)[:, 0])
            ax.plot(np.zeros_like(x_line1)[:, 1], np.array(zp_line1)[:, 1], np.array(zp_line1)[:, 0]); 
            ax.plot(np.zeros_like(x_line1)[:, 1], np.array(zp_line2)[:, 1], np.array(zp_line2)[:, 0])
            ax.scatter(x_int, yx_int, 0)
            ax.scatter(0, yz_int , z_int, )
            ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2])

        
            plt.axis("on")
            plt.show()

    
        return (y_min, y_max),  (xp_line1, xp_line2), (zp_line1, zp_line2) 



    @property
    def __measurement_points(self) -> np.ndarray:
        """
        Return stacked points of X, y and Z

        Where:
            x is the horizontal axis along the plane, 
            y is the depth of measurement point to the origin
            z is the vertical axis along the plae

        """
        cam_xx, cam_zz = np.meshgrid(self.cam_x, self.cam_z)
        points = np.column_stack([cam_xx.flatten("F"), self.camDepth_array.flatten("F"), cam_zz.flatten("F")])
        return points


    def load_Matrix(self, Model:ndarray):
        self.Model = self.cast(Model)
        return True

    @property
    def __hidden_scene_points(self) -> np.ndarray:
        """
        Return stacked points of X, y and Z

        Where:
            x is the horizontal axis along the plane, 
            y is the depth of measurement point to the origin
            z is the vertical axis along the plane

        """

        
        if not self.point_window:
            scene_xx, scene_zz = np.meshgrid(self.scene_x,self.scene_z)
            scene_pixel_locs = np.column_stack([scene_xx.flatten("F"), self.sceneDepth_array.flatten("F"), scene_zz.flatten("F")])

        else:
            x_diff = (self.scene_x[1] - self.scene_x[0]) / (self.point_window+1)
            z_diff = (self.scene_z[1] - self.scene_z[0]) / (self.point_window+1)
            z_len = np.arange(1, self.point_window+1) * z_diff

            x_len = np.arange(1, self.point_window+1) * x_diff
            x = (self.scene_x.reshape((self.N1, 1))*np.ones((self.N1, self.N2))).flatten()
            x = np.tile(x, (self.point_window, 1)).T
            x = x + x_len


            z = np.tile(self.scene_z, (self.point_window, 1)).T
            z = z + z_len
            z = np.tile(z, (self.N1, 1))
          

            self.scenePoints = []

            for i in range(self.N1*self.N2):
                cam_x, cam_z = np.meshgrid(np.array(x[i]), np.array(z[i]))
                scenePoints = np.column_stack([cam_x.flatten("F"), np.zeros(shape=cam_z.flatten("F").shape, dtype=self.default_dtype), cam_z.flatten("F")])
                self.scenePoints.append(scenePoints)

            return np.array(self.scenePoints, dtype="float32")


        return scene_pixel_locs


    def GetMeasurement(self, images:torch.Tensor, Model=None) -> torch.Tensor:
        assert len(images.shape) == 4, f"Expected 4 Dimimension, Got {images.shape} Dim"
        # assert images.shape[1] == self.N1 and images.shape[2] == self.N2 , \
        #     f"Input must be only one image and have shape of BatchSize X {self.N1} X {self.N2} X ChannelSize"
       
        images = self.cast(images)

        if Model is None:
            Model = self.Model

        images = rearrange(images, 'b w h c -> b  (h w) c')
        m = torch.tensordot(a=Model.t(), b=images, dims=((0,), (1,)))
        
        return rearrange(m, '(h w) b c -> b w h c', w=self.M2, h=self.M1)
    


    def GetStandardScene(self, measurement, Model=None):
        shape = measurement.shape
        assert len(shape) == 4, f"Expected 4 Dimimension, Got {shape} Dim"
        assert measurement.shape[1] == self.M1 and measurement.shape[2] == self.M2 , \
            f"Input must be only one image and have shape of {self.M1} X {self.M2} X ChannelSize"
        
        
        measurement = self.cast(measurement)
        if Model is None:
            Model = self.Model
        inverse_model = self.cast(torch.pinverse(Model))
            
        measurement = rearrange(measurement, 'b w h c -> b  (h w) c')
        p = torch.tensordot(a=inverse_model.t(), b=measurement, dims=((0,), (1,)))
        return rearrange(p, '(h w) b c -> b w h c', w=self.N2, h=self.N1)

    def GetTikRegularizedScene(self, measurement:np.ndarray, lambda_reg: float = 0.001, Model=None, shape=None) -> np.ndarray:
        
        shape = (self.N2, self.N1) if shape is None else shape
        
        if Model is None:
            Model = self.Model

        transpose = Model.t()
        
        reg_init = torch.einsum('bj, jk -> bk', transpose, Model)
        I = lambda_reg * torch.eye(transpose.shape[0]).to(Model)
        linv = torch.inverse(reg_init + I)
        
        m = rearrange(measurement, 'b w h c -> b  (h w) c')
        p = torch.matmul(self.cast(linv), torch.matmul(transpose, m))
        
        return rearrange(p, 'b (h w) c -> b w h c', w=shape[0], h=shape[1])
    
    
    def GetTranspose(self, measurements, Model=None, type=None, shape=None, reshape=True):
        
        shape = (self.N2, self.N1) if shape is None else shape
    
        assert len(measurements.shape) == 4, f"Expect 4 Dim, Got {measurements.shape} Dim"

        if Model is None:
            Model = self.Model

        measurements = self.cast(measurements, type)
        
        
        m = rearrange(measurements, 'b w h c -> b  (h w) c')
        p = torch.tensordot(Model, m, dims=((0,), (1,)))
    
        return rearrange(p, '(h w) b c -> b w h c', w=shape[0], h=shape[1]) if reshape else rearrange(p, 'h b c -> b h c')


    def GetCGD(self, measurements, n_iteration=50, Model=None, shape=None, reshape=True):

        measurements = self.cast(measurements, torch.float32)
        shape = (self.N2, self.N1) if shape is None else shape

        assert len( measurements.shape) == 4, f"Expect 4 Dim, Got { measurements.shape} Dim"

        if Model is None:
            Model = self.Model

        Model = self.cast(Model, torch.float32)
        

        p = self.GetTranspose(measurements, Model=Model, type=torch.float32, shape=shape, reshape=False)
    
        ATA = torch.einsum('bj, jk -> bk', Model.t(), Model) 
    
        
        r = p
        x = self.cast(torch.zeros_like(r))

        for _ in range(n_iteration):
            denum = torch.permute(torch.tensordot(ATA, p, dims=((0,), (1,))), [1, 0, 2])

            alpha = torch.divide(torch.square(torch.norm(r, dim=1)), torch.sum(r * denum, dim=1))  
            x = torch.add(x, torch.multiply(alpha[:, None], p))
            r_new = torch.subtract(r, (alpha[:, None] * denum))
            beta = torch.divide(torch.square(torch.norm(r_new, dim=1)), torch.square(torch.norm(r, dim=1)))
            p = torch.add(r_new, torch.multiply(beta[:, None],  p))
            r = r_new
        
        return rearrange(x, 'b (h w) c -> b w h c', w=shape[0], h=shape[1])  if reshape else x



    def GetForwardMatrix(self, x_all:ndarray):


        M = self.M1*self.M2
        shape = x_all.shape
     
        
        p_xx = self.cast(x_all).unsqueeze(1).expand(shape[0], M, shape[1])
        p_xx = p_xx - self.camPoints
        num = p_xx[:, :, 1] * p_xx[:, :, 1]

        matrix = (num/torch.sum(torch.pow(p_xx, 4), dim=-1)).t()
        return matrix


    def GetVisibilityMatrix(self, occluder:ndarray, scenePoints):


        scenePoints = scenePoints.to(self.device)
        N  = scenePoints.shape[0]

        x, y, z, w, h = tuple(occluder)

        D  = self.camDepth
        shape = self.cam_x.shape
        campoints_x = self.cast(np.array([self.cam_x])).expand(N, shape[0])
        campoints_z = self.cast(np.array([self.cam_z])).expand(N, shape[0])


        scenepoints_x = scenePoints[:, 0].unsqueeze(1)
        scenepoints_z = scenePoints[:, 2].unsqueeze(1)
        

        left_edge_x =  x - (w/2.)
        right_edge_x = x + (w/2.)

        left_edge = (D * torch.divide(left_edge_x - scenepoints_x, y)) +  scenepoints_x
        right_edge = (D * torch.divide(right_edge_x - scenepoints_x, y)) +  scenepoints_x

        xss = (torch.where(torch.logical_and(campoints_x >= left_edge, campoints_x <= right_edge), 0., 1.)).unsqueeze(1)


        left_edge_z =  z - (h/2.)
        right_edge_z =  z + (h/2.)

        left_edge = (D * torch.divide(left_edge_z - scenepoints_z, y)) +  scenepoints_z
        right_edge = (D * torch.divide(right_edge_z - scenepoints_z, y)) +  scenepoints_z

        zss = (torch.where(torch.logical_and(campoints_z >= left_edge, campoints_z <= right_edge), 0., 1.)).unsqueeze(1)

        VisMat = 1 - torch.multiply(torch.permute(1-zss, [0, 2, 1]), (1 - xss))
        return self.cast(torch.reshape(torch.permute(VisMat, [2, 1, 0]), [-1, N]))
    


    def GetVisibilityFromBox(self, box):
        
        N  = self.scenePoints.shape[0]

        x, y, z, x1, y1, z1 = tuple(box)

        D  = self.camDepth
        shape = self.cam_x.shape
        campoints_x = self.cast(np.array([self.cam_x])).expand(N, shape[0])
        campoints_z = self.cast(np.array([self.cam_z])).expand(N, shape[0])



        scenepoints_x = self.scenePoints[:, 0, 0].unsqueeze(1)
        scenepoints_z = self.scenePoints[:, 0, 2].unsqueeze(1)


        left_edge = (D * torch.divide(x - scenepoints_x, y)) +  scenepoints_x
        right_edge = (D * torch.divide(x1 - scenepoints_x, y)) +  scenepoints_x

        xss = (torch.where(torch.logical_and(campoints_x >= left_edge, campoints_x <= right_edge), 0., 1.)).unsqueeze(1)


        left_edge = (D * torch.divide(z - scenepoints_z, y)) +  scenepoints_z
        right_edge = (D * torch.divide(z1 - scenepoints_z, y)) +  scenepoints_z

        zss = (torch.where(torch.logical_and(campoints_z >= left_edge, campoints_z <= right_edge), 0., 1.)).unsqueeze(1)

        VisMat = 1 - torch.multiply(torch.permute(1-zss, [0, 2, 1]), (1 - xss))
        return self.cast(torch.reshape(torch.permute(VisMat, [2, 1, 0]), [-1, N]))

     
    def ComputeMatrix(self):


        if not self.point_window:
            
            self.Model  = self.GetForwardMatrix(self.scenePoints)

            if self.occluders:
                assert type(self.occluders) == list or type(self.occluders) == tuple, \
                    "Not in shape"

                for occluder in self.occluders:
                    v = self.GetVisibilityMatrix(occluder, self.scenePoints)
                    self.Model  = self.Model * v

        else:

            self.Model = torch.ones(size = (self.camPoints.shape[0], self.scenePoints.shape[0], self.point_window**2), dtype=self.torch_dtype).to(self.device)
      
            def gridmesh(x_all, i):
            
                A_new = self.GetForwardMatrix(x_all)

                if self.occluders:
                    for occluder in self.occluders:
                        A_new = A_new * self.GetVisibilityMatrix(occluder, x_all)
     
                self.Model[:, i]=A_new
    
            if self.multiprocess:
                Parallel(n_jobs=self.multiprocess, require='sharedmem', prefer="threads")\
                        (delayed(gridmesh)(self.scenePoints[i], i) for i in range(self.N1*self.N2))
            else:

                for i in  range(self.N1*self.N2):
                    gridmesh(self.scenePoints[i], i)

            self.Model = self.Model.mean(-1)



    def get_verts(self, box):
           
            x_min, y_min, z_min, x_max, y_max, z_max = box

            box_vertices = np.array([
                [x_min, y_min, z_min],
                [x_max, y_min, z_min],
                [x_max, y_max, z_min],
                [x_min, y_max, z_min],
                [x_min, y_min, z_max],
                [x_max, y_min, z_max],
                [x_max, y_max, z_max],
                [x_min, y_max, z_max],
            ])

            # define the faces of the box (each one is a triangle)
            box_faces = np.array([
                [0, 1, 2],
                [0, 2, 3],
                [4, 5, 6],
                [4, 6, 7],
                [0, 1, 5],
                [0, 5, 4],
                [2, 3, 7],
                [2, 7, 6],
                [0, 3, 7],
                [0, 7, 4],
                [1, 2, 6],
                [1, 6, 5],
            ],)
            
            return box_vertices, box_faces



    def GetOccluderMesh(self, occluders):
        
        

        mesh = None
        
        for box in occluders:
            
            verts, faces = self.get_verts(box)
            
            cubes = o3d.geometry.TriangleMesh()
            cubes.vertices = o3d.utility.Vector3dVector(verts)
            cubes.triangles = o3d.utility.Vector3iVector(faces)

            if mesh is None:
                mesh = cubes
            else:
                mesh = mesh + cubes

        return mesh.remove_duplicated_triangles()
    
    
    
    def GetForwardModelFromScene(self, scene, batch_size=64, pinhole=False, return_vis=False, scenePoints=None):


        if scenePoints is None:
            scenePoints = self.scenePoints
        
        batch_size = batch_size if len(scenePoints)>batch_size else len(scenePoints)      
        
        vis = []

        
        occluder_scene = o3d.t.geometry.RaycastingScene(nthreads=1)
        cubes = o3d.t.geometry.TriangleMesh.from_legacy(scene)
        
        cube_id = occluder_scene.add_triangles(cubes)
    
        for i in range(0, len(scenePoints), batch_size):
            d, o = self.scenePoints[i:i+batch_size, :], self.camPoints
            d_shape = None
            if len(d.shape) > 2:
                d_shape = (d.shape[0], d.shape[1])
                d = d.view(-1, 3)
    
            direction = -(o[:, None]-d)
            direction = direction/torch.linalg.norm(direction)
            o = o.unsqueeze(1).expand( o.shape[0], len(d), o.shape[1])
            rays =  torch.cat([o, direction], -1)
            
            rays = o3d.core.Tensor(self.cast(rays, torch.float32).cpu().numpy(),
                        dtype=o3d.core.Dtype.Float32)

            ans = occluder_scene.cast_rays(rays)
            sug = (ans["t_hit"]-ans["t_hit"]).numpy()
            sug = self.cast(np.nan_to_num(sug, nan=1.0))
            
            if d_shape is not None:
                sug = sug.reshape(-1, d_shape[0], d_shape[1])
                sug = sug.mean(-1)
            
            #Pinhole version
            if pinhole:
                vis.append(1-sug)
            else:
                vis.append(sug)

        if return_vis:
            return torch.cat(vis, -1)
            
        return self.Model*torch.cat(vis, -1).to(self.device)
            


    def GetPCLocation(self, point_clouds):

        N  = self.scenePoints.shape[0]
        discritization = self.scenePoints.shape[1]
        D = self.camDepth
        point, cood = point_clouds.shape



        if len(self.scenePoints.shape) > 2:
            

            p = self.cast(point_clouds).unsqueeze(0).expand(N*discritization, point, cood)

            scenepoints_x = self.scenePoints[:, :, 0]
            scenepoints_z = self.scenePoints[:, :, 2]

            scenepoints_x = scenepoints_x.flatten().unsqueeze(1)
            scenepoints_z = scenepoints_z.flatten().unsqueeze(1)

            x_points = (D * torch.div(p[:, :, 0] - scenepoints_x, p[:, :, 1])) +  scenepoints_x
            z_points = (D * torch.div(p[:, :, 2] - scenepoints_z, p[:, :, 1])) +  scenepoints_z
            x_points = x_points.reshape(N, discritization, -1)
            z_points = z_points.reshape(N, discritization, -1)


        else:
            p = self.cast(point_clouds).unsqueeze(0).expand(N, point, cood)

         
    
            scenepoints_x = self.scenePoints[:, 0].unsqueeze(1)
            scenepoints_z = self.scenePoints[:, 2].unsqueeze(1)
 

            x_points = (D * torch.div(p[:, :, 0] - scenepoints_x, p[:, :, 1])) +  scenepoints_x
            z_points = (D * torch.div(p[:, :, 2] - scenepoints_z, p[:, :, 1])) +  scenepoints_z

        return [x_points, z_points]
    



class MeshModel(Forward_Model):

   

    def __init__(self, camX_len: list = [0.808, 1.747], camZ_len: list = [0.05, 0.729], 
                 camDepth: float = 1.076, sceneDepth: float = 0, scenePixels: list = [10, 10], 
                 camPixels: list = [50, 50], sceneX_len: list = [0, 0.708], sceneZ_len: list = [0.03, 0.436], 
                 occluders: list = None, multiprocess: int = 0, point_window: int = 8, device: str = "cpu", precision: int = 32) -> None:
        super().__init__(camX_len, camZ_len, camDepth, sceneDepth, scenePixels, camPixels, sceneX_len, sceneZ_len, occluders, multiprocess, point_window, device, precision)


        self.occ_x_min = self.sceneX_len[-1]-0.3
        self.occ_x_max = self.sceneX_len[-1]+0.3
        (self.occ_y_min, self.occ_y_max), _, _ = self.OccMinMax()
        self.occ_z_min = self.sceneZ_len[0]+0.05
        self.occ_z_max = self.sceneZ_len[-1]-0.05

        self.ComputeMatrix()



    def GetMeasurementFromMesh(self, images:torch.Tensor, mesh:trimesh.Trimesh,batch_size=64, return_model:bool=False, ):
    
        shape = images.shape
        assert len(shape) == 4, f"Expected 4 Dimimension, Got {shape} Dim"
        batch = shape[0]
        images = self.cast(images).to(self.device)

        Model = self.GetForwardModelFromMesh(mesh, batch_size=batch_size)
        
        if return_model:
            return self.GetMeasurement(images, Model=Model), Model
        
        return self.GetMeasurement(images, Model=Model)
    

    def ComputeMatrix(self, batch_size=64):
        
        batch_size = batch_size if self.N1*self.N2>batch_size else self.N1*self.N2

        if not self.point_window:
            
            self.Model  = self.GetForwardMatrix(self.scenePoints)

            if self.occluders:
                assert type(self.occluders) == list or type(self.occluders) == tuple, \
                    "Not in shape"
                    
                scene = self.GetOccluderMesh(self.occluders)
                self.Model = self.GetForwardModelFromScene(scene)

                # for occluder in self.occluders:
                #     v = self.GetVisibilityMatrix(occluder, self.scenePoints)
                #     self.Model  = self.Model * v

        else:

            self.Model = torch.ones(size = (self.camPoints.shape[0], self.scenePoints.shape[0], self.point_window**2), dtype=self.torch_dtype).to(self.device)
            
      
            def gridmesh(x_all, i, batch_size=64):
                
                shape = x_all.shape[:2]
                
                x_all = x_all.view(-1, 3)
                model = self.GetForwardMatrix(x_all).reshape( -1, shape[0], shape[1])
             
                
                self.Model[:, i:i+batch_size] = model
                
    
            if self.multiprocess:
                Parallel(n_jobs=self.multiprocess, require='sharedmem', prefer="threads")\
                        (delayed(gridmesh)(self.scenePoints[i:i+batch_size], i, batch_size) for i in range(0, self.N1*self.N2, batch_size))
            else:

                for i in range(0, self.N1*self.N2, batch_size):
                    gridmesh(self.scenePoints[i:i+batch_size], i, batch_size)

            self.Model = self.Model.mean(-1)
            if self.occluders is not None:
                scene = self.GetOccluderMesh(self.occluders)
                self.Model = self.GetForwardModelFromScene(scene)



    def GetForwardModelFromMesh(self, mesh, batch_size=64):
        batch_size = batch_size if self.N1*self.N2>batch_size else self.N1*self.N2
        mesh_o3d = o3d.geometry.TriangleMesh()
        mesh_o3d.vertices = o3d.utility.Vector3dVector(mesh.vertices)
        mesh_o3d.triangles = o3d.utility.Vector3iVector(mesh.faces)
        mesh_o3d.vertex_normals = o3d.utility.Vector3dVector(mesh.vertex_normals)
    
        scene = o3d.t.geometry.RaycastingScene()
        cube = o3d.t.geometry.TriangleMesh.from_legacy(mesh_o3d)
        cube_id = scene.add_triangles(cube)
        
        vis = []
        
        for i in range(0, self.N1*self.N2, batch_size):
            d, o = self.scenePoints[i:i+batch_size, 0], self.camPoints
            
            direction = -(o[:, None]-d)
            direction = direction/torch.linalg.norm(direction)
            o = o.unsqueeze(1).expand( o.shape[0], len(d), o.shape[1])
            rays =  torch.cat([o, direction], -1)
            
            rays = o3d.core.Tensor(rays.cpu().numpy(),
                        dtype=o3d.core.Dtype.Float32)

            ans = scene.cast_rays(rays)
            sug = (ans["t_hit"]-ans["t_hit"]).numpy()
            sug = np.nan_to_num(sug, nan=1.0)
            vis.append(self.cast(sug))
            
        return self.Model*torch.cat(vis, -1)
    

    def PointDiv(self, x_diff=0.2, y_diff=0.1, z_diff = 0.25, ranges=10, _on_max = False):
        
        
        points = []

        temp = self.occ_x_max
        if _on_max:
            self.occ_x_max = self.camX_len[0]


        y_avg = (self.occ_y_max-self.occ_y_min)/ranges
        x_avg = (self.occ_x_max-self.occ_x_min)/ranges
        z_avg = (self.occ_z_max-self.occ_z_min)/ranges
       
        
        for i in range(ranges):
            
            y = self.occ_y_min + y_avg*i
            
            while y <= self.occ_y_min:
                y+=0.01
            
            while y+y_diff >= self.occ_y_max:
                y-=0.01
                
            ys = (y, y+y_diff)
            
            x_min, x_max = occ_range(self.camDepth, ys, self.sceneX_len, self.camX_len)
            z_min, z_max = occ_range(self.camDepth, ys, self.sceneZ_len, self.camZ_len)
            
            
            for j in range(ranges):
                x = x_min + x_avg*j
                
                while x <= self.occ_x_min:
                    x+=0.01
                
                while x+x_diff >= self.occ_x_max:
                        x-=0.01
                        
                xs = (x, x+x_diff)

                for k in range(ranges):
                    z = z_min + z_avg*k
                    
                    while z <= self.occ_z_min:
                        z+=0.01
                    
                    while z+z_diff >= self.occ_z_max:
                        z-=0.01
                    
                    zs = (z, z+z_diff)
                    
                    points.append([xs[0], ys[0], zs[0], xs[1], ys[1], zs[1]])
                    

        if _on_max:
            self.occ_x_max = temp
      
        return points

            





class Pinspeck(Forward_Model):

    def __init__(self, camX_len: list = [0.808, 1.747], camZ_len: list = [0.05, 0.729], camDepth: float = 1.076, sceneDepth: float = 0, scenePixels: list = [10, 10], 
                camPixels: list = [50, 50], sceneX_len: list = [0, 0.708], sceneZ_len: list = [0.03, 0.436], occluders=None, multiprocess: int = 0, point_window: int = 8, 
                device: str = "cpu", precision: int = 32, num_points=50, cube=False) -> None:
        
        
        super().__init__(camX_len, camZ_len, camDepth, sceneDepth, scenePixels, camPixels, sceneX_len, sceneZ_len, occluders, multiprocess, point_window, device, precision)

        self.points = {}
        self.ids = {}
        
        self.ComputeMatrix()

        self.occ_x_min = self.sceneX_len[-1]-0.3
        self.occ_x_max = self.sceneX_len[-1]+0.3
        (self.occ_y_min, self.occ_y_max), _, _ = self.OccMinMax()
        self.occ_z_min = self.sceneZ_len[0]+0.05
        self.occ_z_max = self.sceneZ_len[-1]-0.05
        self.num_points = num_points
        self.cube = cube
        self.points, _ = self.OccluderPointss(num_points=num_points, cube=cube)
        
        
        
    def PointDiv(self, x_diff=0.15, y_diff=0.1, z_diff = 0.2, ranges=10, _on_max = False):
        
        (y_min, y_max), _, _ = self.OccMinMax()
        points = []

        temp = self.occ_x_max
        if _on_max:
            self.occ_x_max = self.camX_len[0]


        y_avg = (self.occ_y_max-self.occ_y_min)/ranges
        x_avg = (self.occ_x_max-self.occ_x_min)/ranges
        z_avg = (self.occ_z_max-self.occ_z_min)/ranges
       
        
        for i in range(ranges):
            
            y = y_min + y_avg*i
            
            while y <= self.occ_y_min:
                y+=0.01
            
            while y+y_diff >= self.occ_y_max:
                y-=0.01
                
            ys = (y, y+y_diff)
            
            x_min, x_max = occ_range(self.camDepth, ys, self.sceneX_len, self.camX_len)
            z_min, z_max = occ_range(self.camDepth, ys, self.sceneZ_len, self.camZ_len)
            
            
            for j in range(ranges):
                x = x_min + x_avg*j
                
                while x <= self.occ_x_min:
                    x+=0.01
                
                while x+x_diff >= self.occ_x_max:
                        x-=0.01
                        
                xs = (x, x+x_diff)

                for k in range(ranges):
                    z = z_min + z_avg*k
                    
                    while z <= self.occ_z_min:
                        z+=0.01
                    
                    while z+z_diff >= self.occ_z_max:
                        z-=0.01
                    
                    zs = (z, z+z_diff)
                    
                    points.append([xs[0], ys[0], zs[0], xs[1], ys[1], zs[1]])
                    

        if _on_max:
            self.occ_x_max = temp
      
        return points
    
            
            
    def ComputeMatrix(self, batch_size=64):
        
        batch_size = batch_size if self.N1*self.N2>batch_size else self.N1*self.N2

        if not self.point_window:
            
            self.Model  = self.GetForwardMatrix(self.scenePoints)

            if self.occluders:
                assert type(self.occluders) == list or type(self.occluders) == tuple, \
                    "Not in shape"
                    
                scene = self.GetOccluderMesh(self.occluders)
                self.Model = self.GetForwardModelFromScene(scene)

                # for occluder in self.occluders:
                #     v = self.GetVisibilityMatrix(occluder, self.scenePoints)
                #     self.Model  = self.Model * v

        else:

            self.Model = torch.ones(size = (self.camPoints.shape[0], self.scenePoints.shape[0], self.point_window**2), dtype=self.torch_dtype).to(self.device)
            
      
            def gridmesh(x_all, i, batch_size=64):
                
                shape = x_all.shape[:2]
                
                x_all = x_all.view(-1, 3)
                model = self.GetForwardMatrix(x_all).reshape( -1, shape[0], shape[1])
             
                
                self.Model[:, i:i+batch_size] = model
                
    
            if self.multiprocess:
                Parallel(n_jobs=self.multiprocess, require='sharedmem', prefer="threads")\
                        (delayed(gridmesh)(self.scenePoints[i:i+batch_size], i, batch_size) for i in range(0, self.N1*self.N2, batch_size))
            else:

                for i in range(0, self.N1*self.N2, batch_size):
                    gridmesh(self.scenePoints[i:i+batch_size], i, batch_size)

            self.Model = self.Model.mean(-1)
            if self.occluders is not None:
                scene = self.GetOccluderMesh(self.occluders)
                self.Model = self.GetForwardModelFromScene(scene)


    def OccluderPointss(self, num_points=(5, 5, 5), cube=False):

        points = {}
        self.num_points = num_points
        self.cube = cube
        (y_min, y_max), _, _ = self.OccMinMax()
        y_min+=0.1; y_max-=0.1
        self.occ_y_max = y_max
        self.occ_y_min = y_min
        
        occ_x_min, occ_x_max = occ_range(self.camDepth, (y_min, y_max), self.sceneX_len, self.camX_len)
        occ_z_min, occ_z_max = occ_range(self.camDepth, (y_min, y_max), self.sceneZ_len, self.camZ_len)
        
        self.occ_x_max = occ_x_max if self.occ_x_max > occ_x_max else self.occ_x_max
        self.occ_z_max = occ_z_max if self.occ_z_max > occ_z_max else self.occ_z_max
        
        self.occ_x_min = occ_x_min if self.occ_x_min < occ_x_min else self.occ_x_min
        self.occ_z_min = occ_z_min if self.occ_z_min < occ_z_min else self.occ_z_min
        
    
        y_lin = np.linspace(y_min, y_max, num_points[1], dtype=self.default_dtype)
        y_avg = (self.occ_y_max-self.occ_y_min)/num_points[1]
        x_avg = (self.occ_x_max-self.occ_x_min)/num_points[0]
        z_avg = (self.occ_z_max-self.occ_z_min)/num_points[2]
        self.x_avg, self.y_avg, self.z_avg = x_avg, y_avg, z_avg
        
        m =0
        for i in range(num_points[1]):
            

            if not cube:

                y = (y_lin[i], y_lin[i])
                x_max = self.occ_x_max
                x_min = self.occ_x_min
                z_min = self.occ_z_min
                z_max = self.occ_z_max

         
                x_lin= np.linspace(x_min, x_max, num_points[0], dtype=self.default_dtype)
                

                for j in range(num_points[0]):
                    z_lin = np.linspace(z_min, z_max, num_points[-1], dtype=self.default_dtype)
                    for k in range(1, num_points[2]):

                        points[(j, i, k)] = [x_lin[j], y[0], z_lin[k]]
                        self.ids[(j, i, k)] = m
                        m+=1


            else:
                y = y_min + y_avg*i

                x_max = self.occ_x_max
                x_min = self.occ_x_min
                z_min = self.occ_z_min
                z_max = self.occ_z_max

                for j in range(num_points[0]):
                    x = x_min + x_avg*j

                    for k in range(num_points[2]):
                        z = z_min + z_avg*k

                        points[(j, i, k)] = [x, y, z, x_avg, y_avg, z_avg]
                        self.ids[m] = (j, i, k)
                        m+=1

                        
        return points, np.stack(list(points.values()))
    
    

        
    def GetPinspeckModel(self, ids, pinhole=False, return_vis=False):
        
        boxes = []
        
        for occ in ids:
            
            id = self.ids[occ]
            point = self.points[id]
                
            box = point[0], point[1], point[2], point[0]+point[3], point[1]+point[4], point[2]+point[5]
            
            boxes.append(box)
            
        scene = self.GetOccluderMesh(boxes)
        return self.GetForwardModelFromScene(scene, pinhole=pinhole, return_vis=return_vis)
    
    
    
    
    def GetABarModel(self, ids=None, return_vis=False, pinhole=True, return_pin=False):
        
        
        if ids is None:
            ids = self.ids.keys()
        
        
        A_bar = torch.zeros(self.Model.shape[0], self.Model.shape[1]*len(ids))
        sum_A_bar = 0
       
        
        for j, i in enumerate(ids):
            
            n = self.N1*self.N2
        
            id = self.ids[i]
            point = self.points[id]
            
            box = point[0], point[1], point[2], point[0]+point[3], point[1]+point[4], point[2]+point[5]
            
            scene = self.GetOccluderMesh([box])
        
            Model = self.GetForwardModelFromScene(scene, pinhole=pinhole, return_vis=return_vis)
          
            sum_A_bar = sum_A_bar + Model
            
            A_bar[:, n*j:n*j+n] = Model
            
            
        if return_pin:
            return self.cast(A_bar), self.cast(sum_A_bar)
            
        if return_vis:
            A_bar = torch.cat([torch.ones_like(self.Model), -self.cast(A_bar)], -1)
        
        else:
            A_bar = torch.cat([self.Model, -self.cast(A_bar)], -1)
            
        
        
        return A_bar, self.cast(sum_A_bar)



def point_as_occluder(point0, box=None, x=(0.55, 0.7), y=(0.6, 0.7), z=(0.15, 0.35), flip=False):
    x_min, x_max =x; y_min, y_max =y ; z_min, z_max = z
    point0 = torch.Tensor(point0).clone()
    reshape = len(point0.shape) < 3
    if reshape:
        point0 = point0.unsqueeze(0)
    if flip:
        point0 = point0[:, :,  [0, 2, 1]].clone()
        
    minimum = point0.min(1)[0].unsqueeze(1)
    maximum = point0.max(1)[0].unsqueeze(1)
    if box is None:
        val_min = torch.Tensor(np.array([[x_min, y_min, z_min]])).to(point0)
        val_max = torch.Tensor(np.array([[x_max, y_max, z_max]])).to(point0)
        
    else:
        val_min = box[:3].unsqueeze(0).to(point0)
        val_max = box[3:].unsqueeze(0).to(point0)
        
    # print(point0.shape, minimum.shape, maximum.shape, val_min.shape, val_max.shape)

    point0 = ((point0 - minimum)/(maximum-minimum))*(val_max-val_min) + val_min

    if reshape:
        return point0[0]

    return point0


   

if __name__=="__main__":  

    M=(128, 128); N=(32, 32)


    B = Pinspeck(camX_len=[0.808, 1.747], camZ_len=[0.05, 0.05+0.939], camDepth= 1.076, sceneDepth=0, scenePixels=N, 
                        camPixels=M, sceneX_len=[0, .708], sceneZ_len=[0.03, 0.436], occluders=None,
                        multiprocess=0, point_window=0, device="cuda:0", precision=32, cube = True, num_points=[5, 5, 5])


    B.ComputeMatrix()
