import torch
from torch.autograd import Variable
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.optim.lr_scheduler import OneCycleLR
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
from SNLLS.world_model import *

# Modify Saunders et al. Grid Search Methods to use Heuristics of the Bounding Box
def huristic_localization(B, m, possible_locations, mesh, optimizer="cgd", num_line=5):
    
    def get_ratio(point0):
        mesh_min = point0.min(0) #torch.min(point0, axis=0)
        mesh_max = point0.max(0)#(point0, axis=0)
        mesh_dims = mesh_max - mesh_min
        longest_dim_index = np.argmax(mesh_dims)
        return mesh_dims/(mesh_dims[longest_dim_index]+1e-9), mesh_dims
    
    ratio, mesh_dims= get_ratio(np.asarray(mesh.vertices))
    possible_locations = B.PointDiv(x_diff=mesh_dims[0], y_diff=mesh_dims[1], z_diff=mesh_dims[2], ranges=num_line)
    possible_locations = list(reversed(possible_locations))
    
    def CheckBox(box):
    
        md = B.GetVisibilityFromBox(box)
        md = torch.Tensor((np.reshape(md.mean(-1).detach().cpu(), (B.M2, B.M1), "F")))
        h =m.detach().cpu().mean(-1)
        v, ii = torch.topk(1-h.flatten(), 8)#int(B.M1*B.M2*0.01))
        p = torch.ones_like(h.flatten())
        p[ii]=0

        v, i = torch.topk(1-md.flatten(),int(B.M1*B.M2*0.15))
        pmd = torch.ones_like(md.flatten())
        pmd[i]=0

        pdd = pmd+p
        pdd[pdd>0]=1
        pdd[pdd<=0]=0
        return torch.allclose(p,pdd) #torch.any(1-pdd) #

    with torch.no_grad():
        minn, idx = 987, 0
        mdd, m22, sc = 0, 0, 0
        b = 0
        j = 0
        
        for i in range(len(possible_locations)):

            box = np.array(possible_locations[i])
            box =  torch.from_numpy(box).type(torch.float32)
        
            if not CheckBox(box):
                continue

            mesh.vertices = o3d.utility.Vector3dVector(point_as_occluder(np.asarray(mesh.vertices), box).numpy())
            md = B.GetForwardModelFromScene(mesh)
            

            if optimizer=="cgd":
                sc1 = B.GetCGD(m, n_iteration=20, Model=md)

            else:
                sc1 = B.GetTikRegularizedScene(m, lambda_reg=0.1, Model=md)
      
            m2 = B.GetMeasurement(sc1, Model=md)
            loss = torch.nn.functional.mse_loss(m/m.max(), m2/m2.max())
            
            if loss.item()<minn:
                minn = loss.item()
                mdd = md #.cpu()
                m22 = m2 #.cpu()
                b= box #.cpu()
                sc =sc1 #.cpu()
            j+=1
            print("completed", i, "Loss ",minn)
 
        print(f"Projected only {j} boxes from a total of {len(possible_locations)} Boxes") 
        return {"Estimated_Model":mdd.type(torch.float32), "Estimated_Measurement":m22.type(torch.float32), "Estimated_Scene":sc.type(torch.float32), "Estimated_Box":b.type(torch.float32)}


def power_iteration(A, num_iterations=2000, tol=1e-18):
    # Initialize a random vector b based on the number of COLUMNS of A
    b = torch.rand(A.shape[1]).to(A)
    
    # Normalize the vector b to have a norm (length) of 1
    b /= torch.norm(b)
    
    for _ in range(num_iterations):
        # Adjusting the core logic: work on the product A^T A to get the right singular vector
        w = torch.mv(A.t(), torch.mv(A, b))
        
        # Normalize the resulting vector w to produce the next vector b_next
        b_next = w / torch.norm(w)
        
        # Check the difference between the current vector b and b_next
        if torch.norm(b_next - b) < tol:
            break
        
        # Update the current vector b for the next iteration
        b = b_next
    
    # The right singular vector is the final b vector
    v = b
    
    # Compute the largest singular value
    sigma_1 = torch.norm(torch.mv(A, v))
    
    # The left singular vector is Av normalized
    u = torch.mv(A, v) / sigma_1
    
    return sigma_1, u, v


def invkronprodbinary(pbkronff, shape, n_iteration=1000):
    
    # pbkronff = pbkronff.reshape(shape)
    pbkronff = rearrange(pbkronff, '(h w) -> w h', w=shape[0], h=shape[1])
    

  
    [Uv, Sv, Vv] = torch.linalg.svd(pbkronff)
    [Uv, Sv, Vv] = Uv[:, 0], Sv[0], Vv[:, 0]
    
    pb_hat = torch.sqrt(Sv)*Vv
    ff_hat = torch.sqrt(Sv)*Uv
    
    
    
    pb_1 = pb_hat[0]
    
    pb_hat = pb_hat/pb_1
    ff_hat = ff_hat/pb_1

    return ff_hat, pb_hat


def LiftingRecovery(B, measurement, n_iteration=50, svd_iteration=2000):

    channel = measurement.shape[-1]
    N = (B.N2, B.N1)

    N_occluder = len(B.points)
    A_bar, A_bar_sum = B.GetABarModel(list(range(0, N_occluder)))
    
    # def f(x):
    #     measurement - A_bar
    t = B.GetCGD(measurement, Model=A_bar, reshape=False, n_iteration=n_iteration)
    
    

    f_ = []
    b_ =[]
    
    

    for i in range(channel):
        f, br = invkronprodbinary(t[0, :, i], (N[0]*N[1], N_occluder+1), n_iteration=svd_iteration)
        f = np.reshape(f.cpu(), N, "F")
        f_.append(f)
        b_.append(br)


    f_ = np.stack(f_, -1)

    if channel > 2:
        bd = [i for i, j in enumerate(b_[0][1:]) if j>0 and b_[1][i+1]>0 and b_[2][i+1]>0] if channel>1 else [i for i, j in enumerate(b_[0][1:]) if j>0]

    else:

        bd = [i for i, j in enumerate(b_[0][1:]) if j>0 ]

    return f_, bd


def GradientDescentJointREcovery(B, measurement, lam=4., n_iterations=5000, step_loss=7000, early_stop_num=100000, verbose_step=1000, Abar=None, 
                                 early_stop_threshold=1e-7, occluder_vector=None, lambda_reg=None, split_learning=True, use_tikhonov=True, progress=True):
    
    # Define a scalar value for Occluder regularization
    lam = torch.tensor([lam]).to(B.Model)
    measurement = measurement.to(B.Model)
    
    # Calculate the total number of occluders
    N_occluder = len(B.points)
    
    # Get the model with occluders, reshaped to a specific shape
    if Abar is None:
        Abar = B.GetABarModel(list(range(N_occluder)), return_vis=True, pinhole=False, return_pin=1)[0].reshape(-1, N_occluder, (B.N2*B.N1))

    # Initialize occluder coefficients
    if occluder_vector is None:
        occluder_vector = B.cast(torch.rand(N_occluder, device=Abar.device))
    occluder_vector.requires_grad = True


    # Define lambda regularization
    lambda_reg = torch.tensor([1.]).to(B.device)
    lambda_reg.requires_grad = True
    
    # Initialize scene
    if split_learning:
        bias = torch.tensor(np.zeros((1, B.N1, B.N2, 3), np.float32), device=B.device)
        bias.requires_grad = True
    
    # Define the optimizer and learning rate scheduler
    if split_learning:
        opt = Adam([
            {'params': occluder_vector, 'lr': 0.05},
            {'params': bias, 'lr': 0.001},
            {'params': lambda_reg, 'lr': 1}
        ])

    else:
        opt = Adam([
            {'params': occluder_vector, 'lr': 0.05},
            {'params': lambda_reg, 'lr': 1}
        ])
            
    scheduler = StepLR(opt, step_size=step_loss, gamma=0.99)
    # scheduler = CosineAnnealingLR(opt, T_max=10000, eta_min=0.0005)
    
    # To keep track of loss values
    loss_history = []

    # Keep Track of Solution with Least Loss
    max_loss = 890
    
    # Optimization loop
    for i in range(n_iterations):
        opt.zero_grad()
        
        # Compute projection
        p = (Abar*torch.sigmoid(lam*occluder_vector[None,:, None])).sum(1)

        # p = torch.tensordot(Abar.permute(1, 0, 2),  torch.sigmoid(lam*sc), dims=((0,), (0,)))
        
        # Apply ReLU activation, enforces Visibility Binary Values (0, and 1)
        p = torch.relu(p - p.max() + 1) * B.Model

      
        
        # Reconstruct the scene using Tikhonov Regularization
        x = B.GetTikRegularizedScene(measurement, Model=p, lambda_reg=lambda_reg) if use_tikhonov else 0
        # x = B.GetCGD(measurement, Model=p, n_iteration=40)


        # Enforce Positivity and add Bias
        x = torch.relu(bias + x) if split_learning else torch.relu( x )
        
        # Get the measurement (Ax)
        mm = B.GetMeasurement(x / (x.max() + 1e-8), Model=p)

        # Compute the loss
        l = torch.nn.functional.mse_loss(mm / (mm.max() + 1e-8), measurement / (measurement.max() + 1e-8))

        # Compute Derivatives
        l.backward()
        
        # Update parameters
        opt.step()
        scheduler.step()
        
        # Record loss history
        loss_history.append(l.item())
        
        # Update best result
        if l.item() < max_loss:
            final_scene = x.clone().detach()
            final_occluder = occluder_vector.clone().detach()
            final_bias = bias.clone().detach() if split_learning else 0
            final_lambda_reg = lambda_reg.clone().detach()
            max_loss = l.item()
            meas = mm.detach()
            
        # Early stopping condition
        if len(loss_history) > early_stop_num and early_stop_threshold > 0 and (np.mean(loss_history[-early_stop_num:]) - loss_history[-1]) < early_stop_threshold:
            break
        
        if i % verbose_step == 0 and progress:
            print(f"Iteration {i}, Loss: {l.item()}")

    return final_scene.cpu(), final_occluder.cpu(), loss_history, final_lambda_reg.cpu(), final_bias




def GradientDescentREcovery(B, measurement, lam=5., step_loss=5000, early_stop_num=5000, verbose_step=1000, Model=None, 
                                 early_stop_threshold=1e-7, lambda_reg=None, split_learning=True, use_tikhonov=True, progress=True):
    # Define a scalar value for Occluder regularization
    measurement = measurement.to(B.Model)
    

    # Get the model with occluders, reshaped to a specific shape
    if Model is None:
        Model = B.Model # = B.GetABarModel(list(range(N_occluder)), return_vis=True, pinhole=False, return_pin=1)[0].reshape(-1, N_occluder, (B.N2*B.N1))

    # Initialize occluder coefficients



    # Define lambda regularization
    lambda_reg = torch.tensor([1.]).to(B.device)
    lambda_reg.requires_grad = True
    
    # Initialize scene
    if split_learning:
        bias = torch.tensor(np.zeros((1, B.N1, B.N2, 3), np.float32), device=B.device)
        bias.requires_grad = True
    
    # Define the optimizer and learning rate scheduler
    if split_learning:
        opt = Adam([
            {'params': bias, 'lr': 0.001},
            {'params': lambda_reg, 'lr': 1}
        ])

    else:
        opt = Adam([
            {'params': lambda_reg, 'lr': 1}
        ])
            
    scheduler = StepLR(opt, step_size=step_loss, gamma=0.7)
    
    # To keep track of loss values
    loss_history = []

    # Keep Track of Solution with Least Loss
    max_loss = 890
    
    # Optimization loop
    for i in range(100000):
        opt.zero_grad()

      
        
        # Reconstruct the scene using Tikhonov Regularization
        x = B.GetTikRegularizedScene(measurement, Model=Model, lambda_reg=lambda_reg) if use_tikhonov else 0

        # Enforce Positivity and add Bias
        x = torch.relu(bias + x) if split_learning else torch.relu( x )
        
        # Get the measurement (Ax)
        mm = B.GetMeasurement(x / (x.max() + 1e-8), Model=Model)

        # Compute the loss
        l = torch.nn.functional.mse_loss(mm / (mm.max() + 1e-8), measurement / (measurement.max() + 1e-8))

        # Compute Derivatives
        l.backward()
        
        # Update parameters
        opt.step()
        scheduler.step()
        
        # Record loss history
        loss_history.append(l.item())
        
        # Update best result
        if l.item() < max_loss:
            final_scene = x.clone().detach()
            final_bias = bias.clone().detach() if split_learning else 0
            final_lambda_reg = lambda_reg.clone().detach()
            max_loss = l.item()
            
        # Early stopping condition
        if len(loss_history) > early_stop_num and early_stop_threshold > 0 and (np.mean(loss_history[-early_stop_num:]) - loss_history[-1]) < early_stop_threshold:
            break
        
        if i % verbose_step == 0 and progress:
            print(f"Iteration {i}, Loss: {l.item()}")

    return final_scene.cpu(), loss_history, final_lambda_reg.cpu(), final_bias

