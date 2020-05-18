"""Helper functions for simulating LTEM images. 

An assortment of helper functions broadly divided into three sections. 
    1) Simulating images from phase shifts
    2) Helper functions for displaying vector fields
    3) Generating variations of magnetic vortex/skyrmion states

AUTHOR:
Arthur McCray, ANL, Summer 2019.
--------------------------------------------------------------------------------
"""

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import colors
from mpl_toolkits.mplot3d import Axes3D
import sys as sys
sys.path.append("..")
import os
from comp_phase import mansPhi
from microscopes import Microscope
from skimage import io as skimage_io
from TIE_helper import *


# ================================================================= #
#                 Simulating phase shift and images                 #
# ================================================================= #

def sim_images(mphi=None, ephi=None, pscope=None, Dshp=None, del_px = 1, 
    def_val = 0, b0=1e4, add_random=False, path=None):
    """

    Change to passing a pscope instead. 

    """
    print(f'Total fov is {np.shape(mphi)[0]*del_px} nm')
    dy, dx = mphi.shape
    if dy == dx:
        dim = dy
        d2 = dim/2
    else:
        print("Expects square grid.")
        return 0

    Tphi = mphi+ephi

    if add_random:
        ran_phi = np.random.uniform(low = -np.pi/16,
                                    high = np.pi/16,
                                    size=[dim,dim])
        Tphi += ran_phi

    #material parameters for phase and amp. computation.
    #lattice values
    isl_xip0 = 50 #nm
    isl_thk = 20 #nm
    thk2 = isl_thk/del_px #thickness in pixels 
    
    #Support membrane
    mem_thk = 50.0 #nm
    mem_xip0 = 1000.0 #nm 

    line = np.arange(dim)-float(d2)
    [Y,X] = np.meshgrid(line,line)
    qq = np.sqrt(X**2 + Y**2) / float(dim)


    #amplitude
    if Dshp is None:
        thk_map = np.ones(mphi.shape)
    else:
        thk_map = np.sum(mask, axis=2)
    Amp = np.exp((-np.ones([dim,dim]) * mem_thk / mem_xip0) - (thk_map / isl_xip0))
    ObjWave = Amp * (np.cos(Tphi) + 1j * np.sin(Tphi))

    # compute unflipped images
    pscope.defocus = 0.0
    im_in = pscope.getImage(ObjWave,qq,del_px)
    pscope.defocus = -def_val
    im_un = pscope.getImage(ObjWave,qq,del_px)
    pscope.defocus = def_val
    im_ov = pscope.getImage(ObjWave,qq,del_px)
    
    if path is not None:
        print(f'saving to {path}')
        im_stack = np.array([im_un, im_in, im_ov])

        if not os.path.exists(path):
                os.makedirs(path)

        skimage_io.imsave(path + 'align.tiff', im_stack.astype('float32'),plugin='tifffile')
        skimage_io.imsave(path + 'phase_m.tiff', mphi.astype('float32'),plugin='tifffile')

        with open (path + 'params.txt', 'w') as text:
            text.write("defocus, del_px, EE, im_size (pix), thickness (nm), b0\n")
            text.write(f"{def_val}\n")
            text.write(f"{del_px}\n") 
            text.write(f"{pscope.E}\n")        
            text.write(f"{dim}\n")
            text.write(f"{isl_thk}\n") 
            text.write(f"{b0}\n") 

    return (Tphi, im_ov, im_in, im_un)

def std_mansPhi(mag_x, mag_y, shape=None, del_px = 1, b0=1e4):
    #material parameters for phase and amp. computation.
    #lattice values
    isl_V0 = 20 #V inner potential default 20
    isl_thk = 20 #nm thickness default 20
    thk2 = isl_thk/del_px #thickness in pixels 

    #Magnetization parameters
    phi0 = 2.07e7 #Gauss*nm^2 
    cb = b0/phi0*del_px**2 #1/px^2

    #Support membrane
    mem_thk = 0.0 #nm
    mem_V0 = 10 #V default 10

    # actually calculate magnetic phase shift with mansuripur algorithm
    mphi = mansPhi(bx = mag_x, by = mag_y, thick = thk2)*np.pi*2*cb

    # and now electric phase shift
    if shape is None:
        thk_map = np.ones(mag_x.shape)
    else:
        thk_map = shape 
    pscope = Microscope(E=200e3,Cs = 200.0e3, theta_c = 0.01e-3, def_spr = 80.0)
    ephi = pscope.sigma * (thk_map * isl_V0 + np.ones(mag_x.shape) * mem_thk * mem_V0)
    
    return (ephi, mphi)

# ================================================================= #
#           Various functions for displaying vector fields          #
# ================================================================= #

def show_3D(mag_x, mag_y, mag_z, a = 15, show_all = True, l = None, za = 15):
    '''
    a is number of arrows in x direction (y direction scaled automatically)
    za is number of arrows in z direction
    '''
    
    a = ((mag_x.shape[0] - 1)//a)+1
    bmax = max(mag_x.max(), mag_y.max(),mag_z.max())

    if l is None:
        l = mag_x.shape[0]/(2*bmax*a)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    dimx = mag_x.shape[0]
    dimy = mag_x.shape[1]
    if mag_x.ndim == 3:
        dimz = mag_x.shape[2]
        if za > dimz:
            za = 1
        else:
            za = ((mag_x.shape[2] - 1)//za)+1
        
        X,Y,Z = np.meshgrid(np.arange(0,dimx,1),
                       np.arange(0,dimy,1),
                       np.arange(0,dimz*a,a))
    else:
        dimz = 1
        X,Y,Z = np.meshgrid(np.arange(0,dimx,1),
                       np.arange(0,dimy,1),
                       np.arange(0,1,1))
    
    # doesnt handle (0,0,0) arrows very well, so this puts in very small ones. 
    zeros = mag_x.astype('bool')+mag_y.astype('bool')+mag_z.astype('bool')
    mag_z[np.where(~zeros)] = bmax/100000
    mag_x[np.where(~zeros)] = bmax/100000
    mag_y[np.where(~zeros)] = bmax/100000

    U = mag_x.reshape((dimx,dimy,dimz))
    V = mag_y.reshape((dimx,dimy,dimz))
    W = mag_z.reshape((dimx,dimy,dimz))

    # maps in plane direction to hsv wheel, out of plane to white (+z) and black (-z)
    phi = np.ravel(np.arctan2(V[::a,::a,::za],U[::a,::a,::za]))
    # map phi from [pi,-pi] -> [1,0]
    hue = phi/(2*np.pi)+0.5

    # setting the out of plane values now
    theta = np.arctan2(W[::a,::a,::za],(U[::a,::a,::za]**2+V[::a,::a,::za]**2))
    value = np.ravel(np.where(theta<0, 1+2*theta/np.pi, 1))
    sat = np.ravel(np.where(theta>0, 1-2*theta/np.pi, 1))

    arrow_colors = np.squeeze(np.dstack((hue, sat, value)))
    arrow_colors = colors.hsv_to_rgb(arrow_colors)

    if show_all: # all alpha values one
        alphas = np.ones((np.shape(arrow_colors)[0],1))
    else: #alpha values map to inplane component
        alphas = np.minimum(value, sat).reshape(len(value),1)
        value = np.ones(value.shape)
        sat = np.ravel(1-abs(2*theta/np.pi))
        arrow_colors = np.squeeze(np.dstack((hue, sat, value)))
        arrow_colors = colors.hsv_to_rgb(arrow_colors)

        ax.set_facecolor('black')
        ax.w_xaxis.set_pane_color((0, 0, 0, 1.0))
        ax.w_yaxis.set_pane_color((0, 0, 0, 1.0))
        ax.w_zaxis.set_pane_color((0, 0, 0, 1.0))
        # ax.xaxis.pane.set_edgecolor('w')
        # ax.yaxis.pane.set_edgecolor('w')
        ax.grid(False)

    # add alpha value to rgb list 
    arrow_colors = np.array([np.concatenate((arrow_colors[i], alphas[i])) for i in range(len(alphas))])
    # quiver colors shaft then points: for n arrows c=[c1, c2, ... cn, c1, c1, c2, c2, ...]
    arrow_colors = np.concatenate((arrow_colors,np.repeat(arrow_colors,2, axis=0))) 

    q = ax.quiver(X[::a,::a,::za], Y[::a,::a,::za], Z[::a,::a,::za], 
                  U[::a,::a,::za], V[::a,::a,::za], W[::a,::a,::za],
                  color = arrow_colors, 
                  length= float(l), 
                  pivot = 'middle', 
                  normalize = False)

    ax.set_xlim(0,dimx)
    ax.set_ylim(0,dimy)
    if dimz == 1:
        ax.set_zlim(-dimx//2, dimx//2)
    else:
        # ax.set_zlim(0, dimz*a)
        ax.set_zlim(-dimx//2 + dimz, dimx//2 + dimz)


    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.show()
    
def show_2D(mag_x, mag_y, a = 15, l = None, title = None):
    a = ((mag_x.shape[0] - 1)//a)+1
    bmax = max(mag_x.max(), mag_y.max())
    if l is None:
        l = mag_x.shape[0]/(10*bmax*a)


    dim = mag_x.shape[0]
    X = np.arange(0, dim, 1)
    Y = np.arange(0, dim, 1)
    U = mag_x 
    V = mag_y 
    
    fig, ax = plt.subplots()
    q = ax.quiver(X[::a], Y[::a], U[::a,::a], V[::a,::a], 
                  units='inches', 
                  scale = l,
                  pivot = 'mid')
    # qk = ax.quiverkey(q, 0.9, 0.9, 2, r'$=$'+str(l), labelpos='E',
    #                    coordinates='figure')
    if title is not None:
        ax.set_title(title)
    ax.set_aspect(1)

    plt.show()


def show_sims(phi, im_ov, im_in, im_un):
    vmax = np.max(phi)+.05
    vmin = np.min(phi)-.05
    fig = plt.figure(figsize=(12,3))
    ax11 = fig.add_subplot(141)
    im = ax11.imshow(phi,cmap='gray', origin = 'upper', vmax = vmax, vmin = vmin)
    plt.axis('off')
    plt.title('Phase')
    vmax = np.max(im_un) + .05
    vmin = np.min(im_un) - .05
    ax = fig.add_subplot(142)
    ax.imshow(im_un,cmap='gray', origin = 'upper', vmax = vmax, vmin = vmin)
    plt.axis('off')
    plt.title('Underfocus')
    ax2 = fig.add_subplot(143)
    ax2.imshow(im_in,cmap='gray', origin = 'upper', vmax = vmax, vmin = vmin)
    plt.axis('off')
    plt.title('In-focus')
    ax3 = fig.add_subplot(144)
    ax3.imshow(im_ov,cmap='gray', origin = 'upper', vmax = vmax, vmin = vmin)
    plt.axis('off')
    plt.title('Overfocus')



# ================================================================= #
#                                                                   #
#                Making vortex magnetization states                 #
#                                                                   #
# ================================================================= #

def Lillihook(dim, rad = None, Q = 1, gamma = np.pi/2, P=1, show=False): 
    ''' Make a skyrmion with magnetizations defined from Lilliehook PRB 1997
    dim: the dimension of the final (square) arrays
    rad: a length parameter for the skyrmion
    Q: topological charge: 1 for skyrmion, 2 for biskyrmion, -1 for antiskyrmion
    gamma: helicity; 0 or pi = Neel; 3pi/2 or pi/2 = Bloch. 
    P: polarity (z direction in center)
    '''

    cx, cy = [dim//2,dim//2] 
    cy = dim//2
    cx = dim//2      
    if rad is None:
        rad = dim//16
        print(f'Rad set to {rad}.')
    a = np.arange(dim)
    b = np.arange(dim)
    x,y = np.meshgrid(a,b)
    x -= cx
    y -= cy
    dist = np.sqrt(x**2 + y**2)
    zeros = np.where(dist==0)
    dist[zeros] = 1

    f = ((dist/rad)**(2*Q)-4) / ((dist/rad)**(2*Q)+4)

    re = np.real(np.exp(1j*gamma))
    im = np.imag(np.exp(1j*gamma))

    mag_x = -np.sqrt(1-f**2) * (re*np.cos(Q*np.arctan2(y,x)) + im*np.sin(Q*np.arctan2(y,x)))
    mag_y = -np.sqrt(1-f**2) * (-1*im*np.cos(Q*np.arctan2(y,x)) + re*np.sin(Q*np.arctan2(y,x)))

    mag_z = -P*f
    mag_x[zeros] = 0
    mag_y[zeros] = 0

    if show:
        # show_im(np.sqrt(mag_x**2 + mag_y**2 + mag_z**2), 'mag')
        show_im(mag_x, 'mag x')
        show_im(mag_y, 'mag y')
        show_im(mag_z, 'mag z')
        
        x = np.arange(0,dim,1)
        fig,ax = plt.subplots()
        ax.plot(x,mag_z[dim//2], label='z')
        ax.set_xlabel("x axis, y=0")
        ax.set_ylabel("mag_z")
        # plt.legend()
        plt.show()

    return (mag_x, mag_y, mag_z)


def Bloch(dim, chirality = 'cw', pad = True, ir=0, show=False): 
    cx, cy = [dim//2,dim//2]
    if pad: 
        rad = 3*dim//8
    else:
        rad = dim//2
        
    # mask
    x,y = np.ogrid[:dim, :dim]
    cy = dim//2
    cx = dim//2
    r2 = (x-cx)*(x-cx) + (y-cy)*(y-cy)
    circmask = r2 <= rad*rad
    circmask *= r2 >= ir*ir

    # making the magnetizations
    a = np.arange(dim)
    b = np.arange(dim)
    x,y = np.meshgrid(a,b)
    x -= cx
    y -= cy
    dist = np.sqrt(x**2 + y**2)
    
    mag_x = -np.sin(np.arctan2(y,x)) * np.sin(np.pi*dist/(rad-ir) - np.pi*(2*ir-rad)/(rad-ir)) * circmask
    mag_y =  np.cos(np.arctan2(y,x)) * np.sin(np.pi*dist/(rad-ir) - np.pi*(2*ir-rad)/(rad-ir)) * circmask
    mag_x /= np.max(mag_x)
    mag_y /= np.max(mag_y)
    
    mag_z = (-ir-rad + 2*dist)/(ir-rad) * circmask
    mag_z[np.where(dist<ir)] = 1
    mag_z[np.where(dist>rad)] = -1

    mag = np.sqrt(mag_x**2 + mag_y**2 + mag_z**2)
    mag_x /= mag 
    mag_y /= mag 
    mag_z /= mag

    if chirality == 'ccw':
        mag_x *= -1
        mag_y *= -1
    
    if show:
        show_im(np.sqrt(mag_x**2 + mag_y**2 + mag_z**2), 'mag')
        show_im(mag_x, 'mag x')
        show_im(mag_y, 'mag y')
        show_im(mag_z, 'mag z')
        
        x = np.arange(0,dim,1)
        fig,ax = plt.subplots()
        ax.plot(x,mag_x[dim//2],label='x')
        ax.plot(x,-mag_y[:,dim//2],label='y')
        ax.plot(x,mag_z[dim//2], label='z')
        plt.legend()
        plt.show()

    return (mag_x, mag_y, mag_z)


def Neel(dim, chirality = 'io', pad = True, ir=0,show=False):
    cx, cy = [dim//2,dim//2]
    if pad: 
        rad = 3*dim//8
    else:
        rad = dim//2

    # mask
    x,y = np.ogrid[:dim, :dim]
    cy = dim//2
    cx = dim//2
    r2 = (x-cx)*(x-cx) + (y-cy)*(y-cy)
    circmask = r2 <= rad*rad
    circmask *= r2 >= ir*ir

    # making the magnetizations
    a = np.arange(dim)
    b = np.arange(dim)
    x,y = np.meshgrid(a,b)
    x -= cx
    y -= cy
    dist = np.sqrt(x**2 + y**2)

    mag_x = -x * np.sin(np.pi*dist/(rad-ir) - np.pi*(2*ir-rad)/(rad-ir)) * circmask
    mag_y = -y * np.sin(np.pi*dist/(rad-ir) - np.pi*(2*ir-rad)/(rad-ir)) * circmask
    mag_x /= np.max(mag_x)
    mag_y /= np.max(mag_y)

    # b = 1
    # mag_z = (b - 2*b*dist/rad) * circmask
    mag_z = (-ir-rad + 2*dist)/(ir-rad) * circmask

    mag_z[np.where(dist<ir)] = 1
    mag_z[np.where(dist>rad)] = -1

    mag = np.sqrt(mag_x**2 + mag_y**2 + mag_z**2)
    mag_x /= mag 
    mag_y /= mag 
    mag_z /= mag

    if chirality == 'oi':
        mag_x *= -1
        mag_y *= -1

    if show:
        show_im(np.sqrt(mag_x**2 + mag_y**2 + mag_z**2), 'mag')
        show_im(mag_x, 'mag x')
        show_im(mag_y, 'mag y')
        show_im(mag_z, 'mag z')
        
        x = np.arange(0,dim,1)
        fig,ax = plt.subplots()
        ax.plot(x,mag_x[dim//2],label='x')
        ax.plot(x,-mag_y[:,dim//2],label='y')
        ax.plot(x,mag_z[dim//2], label='z')

        plt.legend()
        plt.show()

    return (mag_x, mag_y, mag_z)