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
import textwrap


# ================================================================= #
#                 Simulating phase shift and images                 #
# ================================================================= #

def sim_images(mphi=None, ephi=None, pscope=None, isl_shape=None, del_px=1, 
    def_val=0, add_random=False, save_path=None,
    isl_thk=20, isl_xip0=50, mem_thk=50, mem_xip0=1000):
    """Simulate LTEM images for a given electron phase shift through a sample. 

    This function returns LTEM images simulated for in-focus and at +/- def_val 
    for comparison to experimental data and reconstruction. 

    It was primarily written for simulating images of magnetic island 
    structures, and as such the sample is defined in two parts: a uniform 
    support membrane across the region and islands of magnetic material defined
    by an array isl_shape. The magnetization is defined with 2D arrays 
    corresponding to the x- and y-components of the magnetization vector. 

    There are many required parameters here that must be set to account for the 
    support membrane. The default values apply to 20nm permalloy islands on a 
    50nm SiN membrane window. 

    Args:
        mphi: 2D array (M, N). magnetic phase shift
        ephi: 2D array (M, N). Electrostatic phase shift
        pscope: Microscope object. Set appropriate parameters. 
        isl_shape: 2D or 3D float array (M, N, Z). If 2D the thickness will be
            taken as the isl_shape values multiplied by isl_thickness. If 3D, the 
            isl_shape array will be summed along the z access becoming 2D. 
            Default None -> uniform flat material with thickness = isl_thk. 
        del_px: Float. Scale factor (nm/pixel). Default = 1. 
        def_val: Float. The defocus values at which to calculate the images .
        add_random: Bool or Float. Whether or not to add amorphous background to
            the simulation. True or 1 will add the default background, any 
            other value will be multiplied to scale the additional phase term. 
        save_path: String. Will save a stack [underfocus, infocus, overfocus] as
            well as (mphi+ephi) as tiffs along with a params.text file. 
            Default None: Does not save. 
    Material Parameter Args:
        isl_thk: Float. Island thickness (nm). Default 20. 
        isl_xip0: Float. Island extinction distance (nm). Default 50. 
        mem_thk: Float. Support membrane thickness (nm). Default 50. 
        mem_xip0: Float. Support membrane extinction distance (nm). Default 1000. 

    Returns: (Tphi, im_un, im_in, im_ov)
        Tphi: 2D array (M,N). Total electron phase shift (ephi+mphi).
        im_un: 2D array (M,N). Simulated image at delta z = -def_val.
        im_in: 2D array (M,N). Simulated image at zero defocus.
        im_ov: 2D array (M,N). Simulated image at delta z = +def_val.
    """
    Tphi = mphi + ephi
    print(f'Total fov is ({np.shape(Tphi)[1]*del_px},{np.shape(Tphi)[0]*del_px}) nm')
    dy, dx = Tphi.shape

    if add_random:
        if type(add_random) == bool:
            add_random = 1.0
        ran_phi = np.random.uniform(low = -np.pi/128*add_random,
                                    high = np.pi/128*add_random,
                                    size=[dy,dx])
        ran_phi *= np.max(ephi)
        Tphi += ran_phi

    thk2 = isl_thk/del_px #thickness in pixels 
    qq = dist(dy, dx, shift=True)

    #amplitude
    if isl_shape is None:
        thk_map = np.ones(Tphi.shape)*isl_thk
    else:
        if type(isl_shape) != np.ndarray:
            isl_shape = np.array(isl_shape)
        if isl_shape.ndim == 3:
            thk_map = np.sum(isl_shape, axis=2)*isl_thk
        elif isl_shape.ndim == 2:
            thk_map = isl_shape*isl_thk
        else:
            print(textwrap.dedent(f"""
                Mask given must be 2D (y,x) or 3D (y,x,z) array. 
                It was given as a {isl_shape.ndim} dimension array."""))
            sys.exit(1)

    Amp = np.exp((-np.ones([dy,dx]) * mem_thk / mem_xip0) - (thk_map / isl_xip0))
    ObjWave = Amp * (np.cos(Tphi) + 1j * np.sin(Tphi))

    # compute unflipped images
    pscope.defocus = 0.0
    im_in = pscope.getImage(ObjWave,qq,del_px)
    pscope.defocus = -def_val
    im_un = pscope.getImage(ObjWave,qq,del_px)
    pscope.defocus = def_val
    im_ov = pscope.getImage(ObjWave,qq,del_px)
    
    if save_path is not None:
        print(f'saving to {save_path}')
        im_stack = np.array([im_un, im_in, im_ov])
        if not os.path.exists(save_path):
                os.makedirs(save_path)
        skimage_io.imsave(save_path + 'align.tiff', im_stack.astype('float32'),plugin='tifffile')
        skimage_io.imsave(save_path + 'phase.tiff', Tphi.astype('float32'),plugin='tifffile')

        with open (save_path + 'params.txt', 'w') as text:
            text.write("defocus, del_px, EE, im_size (pix), thickness (nm)\n")
            text.write(f"def_val\t{def_val}\n")
            text.write(f"del_px\t{del_px}\n") 
            text.write(f"scope En.\t{pscope.E}\n")        
            text.write(f"ims_size\t({dy},{dx})\n")
            text.write(f"isl_thk\t{isl_thk}\n") 
            text.write(f"isl_xip0\t{isl_xip0}\n") 
            text.write(f"mem_thk\t{mem_thk}\n") 
            text.write(f"mem_xip0\t{mem_xip0}\n") 
            text.write(f"add_random\t{add_random}\n") 

    return (Tphi, im_un, im_in, im_ov)


def std_mansPhi(mag_x=None, mag_y=None, del_px = 1, isl_shape=None, pscope=Microscope(),
    b0=1e4, isl_thk=20, isl_V0=20, mem_thk=50, mem_V0=10):
    """Calculates the electron phase shift through a given 2D magnetization. 
    
    This function was originally created for simulating LTEM images of magnetic
    island structures, and it is kept as an example of how to set up and use the
    mansPhi function. It defines the sample in two parts: a uniform membrane 
    across the region and an island structure defined by isl_shape. The 
    magnetization is defined with 2D arrays corresponding to the x- and y-
    components of the magnetization vector. 

    The magnetic phase shift is calculated using the Mansuripur algorithm (see 
    comp_phase.py), and the electrostatic phase shift is computed directly from
    the materials parameters and geometry given. 
    
    Args:
        mag_x: 2D Array. X-component of the magnetization at each pixel. 
        mag_y: 2D Array. Y-component of the magnetization at each pixel. 
        isl_shape: 2D or 3D float array shape (M, N, Z). If 2D the thickness will
            be taken as the isl_shape values multiplied by isl_thickness. If 3D, 
            the isl_shape array will be summed along the z access becoming 2D. 
            Default = None -> uniform flat material with thickness = isl_thk.
        del_px: Float. Scale factor (nm/pixel). Default = 1. 
    Material Parameter Args: 
        pscope: Microscope object. Accelerating voltage is the relevant parameter.
        b0: Float. Saturation magnetization (gauss). Default 1e4.
        isl_thk: Float. Island thickness (nm). Default 20. 
        isl_V0: Float. Island mean inner potential (V). Default 20. 
        mem_thk: Float. Support membrane thickness (nm). Default 50. 
        mem_V0: Float. Support membrane mean inner potential (V). Default 10. 
        
    Returns: (ephi, mphi)
        ephi: Electrostatic phase shift, 2D array
        mphi: magnetic phase shift, 2D array
    """
    thk2 = isl_thk/del_px #thickness in pixels 
    phi0 = 2.07e7 #Gauss*nm^2 flux quantum
    cb = 2*np.pi*b0/phi0*del_px**2 #1/px^2

    # calculate magnetic phase shift with mansuripur algorithm
    mphi = mansPhi(bx=mag_x, by=mag_y, thick = thk2)*cb

    # and now electric phase shift
    if isl_shape is None:
        thk_map = np.ones(mag_x.shape)*isl_thk
    else:
        if type(isl_shape) != np.ndarray:
            isl_shape = np.array(isl_shape)
        if isl_shape.ndim == 3:
            thk_map = np.sum(isl_shape, axis=2)*isl_thk
        elif isl_shape.ndim == 2:
            thk_map = isl_shape*isl_thk
        else:
            print(textwrap.dedent(f"""
                Mask given must be 2D (y,x) or 3D (y,x,z) array. 
                It was given as a {isl_shape.ndim} dimension array."""))
            sys.exit(1)

    ephi = pscope.sigma * (thk_map * isl_V0 + np.ones(mag_x.shape) * mem_thk * mem_V0)
    return (ephi, mphi)

# ================================================================= #
#           Various functions for displaying vector fields          #
# ================================================================= #

# These display functions were largely hacked together, and any advice or 
# resources for improving/replacing them (while still working within jupyter 
# notebooks) would be greatly appreciated. Contact: AMcCray@anl.gov

def show_3D(mag_x, mag_y, mag_z, a=15, ay=None, az=15, l=None, show_all=True):
    """ Display a 3D vector field with arrows. 

    Arrow color is determined by direction, with in-plane mapping to a HSV 
    colorwheel and out of plane to white (+z) and black (-z). 

    Plot can be manipulated by clicking and dragging with the mouse. a, ay, and 
    az control the  number of arrows that will be plotted along each axis, i.e. 
    there will be a*ay*az total arrows. In the default case a controls both ax 
    and ay. 

    Args: 
        mag_x: 2D array. x-component of magnetization. 
        mag_y: 2D array. y-component of magnetization. 
        mag_z: 2D array. z-component of magnetization. 
        a: int. Number of arrows to plot along the x-axis, if ay=None then this 
            sets the y-axis too. 
        ay: int. Number of arrows to plot along y-axis. Defaults to a. 
        az: int. Number of arrows to plot along z-axis. if az > depth of array, 
            az is set to 1. 
        l: Float. Scale of arrows. Larger -> longer arrows. 
        show_all: Bool. 
            True: All arrows are displayed with a grey background. 
            False: Alpha value of arrows is controlled by in-plane component. 
                As arrows point out-of-plane they become transparent, leaving 
                only in-plane components visible. The background is black. 

    Returns: None
        Displays a matplotlib axes3D object. 
    """ 
    if ay is None:
        ay = a
    ay = ((mag_x.shape[0] - 1)//a)+1
    axx = ((mag_x.shape[1] - 1)//a)+1

    bmax = max(mag_x.max(), mag_y.max(),mag_z.max())

    if l is None:
        l = mag_x.shape[0]/(2*bmax*a)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    dimx = mag_x.shape[0]
    dimy = mag_x.shape[1]
    if mag_x.ndim == 3:
        dimz = mag_x.shape[2]
        if az > dimz:
            az = 1
        else:
            az = ((mag_x.shape[2] - 1)//az)+1
        
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
    phi = np.ravel(np.arctan2(V[::ay,: :axx,::az],U[::ay,: :axx,::az]))
    # map phi from [pi,-pi] -> [1,0]
    hue = phi/(2*np.pi)+0.5

    # setting the out of plane values now
    theta = np.arctan2(W[::ay,: :axx,::az],(U[::ay,: :axx,::az]**2+V[::ay,: :axx,::az]**2))
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

    q = ax.quiver(X[::ay,: :axx,::az], Y[::ay,: :axx,::az], Z[::ay,: :axx,::az], 
                  U[::ay,: :axx,::az], V[::ay,: :axx,::az], W[::ay,: :axx,::az],
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
    """ Display a 2D vector arrow plot. 

    Args: 
        mag_x: 2D array. x-component of magnetization. 
        mag_y: 2D array. y-component of magnetization. 
        a: int. Number of arrows to plot along the x and y axes. Default 15. 
        l: Float. Scale factor of arrows. Larger -> longer arrows. Default None
            makes a guess at a good value. 
        title: String. Title for plot. Default None. 

    Returns: None
        Displays matplotlib plot. 
    """ 
    a = ((mag_x.shape[0] - 1)//a)+1
    bmax = max(mag_x.max(), mag_y.max())
    if l is None: # approximating something that might work. 
        l = mag_x.shape[0]/(5*bmax*a)
    else:
        l = 20/l

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
    if title is not None:
        ax.set_title(title)
    ax.set_aspect(1)
    plt.show()
    return


def show_sims(phi, im_un, im_in, im_ov):
    """Plot phase, underfocus, infocus, and overfocus images in one plot.
    
    Uses same scale of intensity values for all simulated images but not phase. 

    Args:
        phi: 2D array. Image of phase shift of object (or anything). 
        im_un: 2D array. Underfocus image. 
        im_in: 2D array. Infocus image. 
        im_ov: 2D array. Overfocus image. 
    
    Returns: None
        Displays matplotlib plot. 
    """
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
    return


# ================================================================= #
#                                                                   #
#                Making vortex magnetization states                 #
#                                                                   #
# ================================================================= #

def Lillihook(dim, rad = None, Q = 1, gamma = np.pi/2, P=1, show=False): 
    """Define a skyrmion magnetization. 

    This function makes a skyrmion magnetization as calculated and defined in 
    [1]. It returns three 2D arrays of size (dim, dim) containing the x, y, and 
    z magnetization components at each pixel. 

    Args: 
        dim: Int. Dimension of lattice. 
        rad: Float. Radius parameter (see [1]). Default dim//16. 
        Q: Int. Topological charge. 
            1: skyrmion
            2: biskyrmion
            -1: antiskyrmion
        gamma: Float. Helicity. 
            0 or Pi: Neel
            Pi/2 or 3Pi/2: Bloch
        P: Polarity (z direction in center)
        show: Bool: If True, will show the x, y, z components in plot form. 

    Returns: (mag_x, mag_y, mag_z)
        mag_x: 2D array (dim, dim). x-component of magnetization vector. 
        mag_y: 2D array (dim, dim). y-component of magnetization vector. 
        mag_z: 2D array (dim, dim). z-component of magnetization vector. 

    References: 
    [1] Lilliehöök, D., Lejnell, K., Karlhede, A. & Sondhi, S. 
        Quantum Hall Skyrmions with higher topological charge. Phys. Rev. B 56, 
        6805–6809 (1997).
    """

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
        show_im(mag_x, 'mag x')
        show_im(mag_y, 'mag y')
        show_im(mag_z, 'mag z')
        x = np.arange(0,dim,1)
        fig,ax = plt.subplots()
        ax.plot(x,mag_z[dim//2], label='mag_z profile along x axis.')
        ax.set_xlabel("x axis, y=0")
        ax.set_ylabel("mag_z")
        plt.legend()
        plt.show()
    return (mag_x, mag_y, mag_z)


def Bloch(dim, chirality = 'cw', pad = True, ir=0, show=False): 
    """Create a BLoch vortex magnetization structure. 

    Unlike Lillihook, this function does not produce a rigorously calculated 
    magnetization structure, but rather one that looks like some experimental 
    observations. 

    Args: 
        dim: Int. Dimension of lattice. 
        chirality: String. 
            'cw': clockwise rotation
            'ccw': counter-clockwise rotation. 
        pad: Bool. Whether or not to leave some space between the edge of the 
            magnetization and the edge of the image. 
        ir: Float. Inner radius of the vortex in pixels. 
        show: Bool: If True, will show the x, y, z components in plot form. 

    Returns: (mag_x, mag_y, mag_z)
        mag_x: 2D array (dim, dim). x-component of magnetization vector. 
        mag_y: 2D array (dim, dim). y-component of magnetization vector. 
        mag_z: 2D array (dim, dim). z-component of magnetization vector. 
    """
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
        show_im(mag_x, 'mag x')
        show_im(mag_y, 'mag y')
        show_im(mag_z, 'mag z')
        x = np.arange(0,dim,1)
        fig,ax = plt.subplots()
        ax.plot(x,mag_z[dim//2], label='mag_z profile along x axis.')
        plt.legend()
        plt.show()
    return (mag_x, mag_y, mag_z)


def Neel(dim, chirality = 'io', pad = True, ir=0,show=False):
    """Create a neel magnetization structure. 

    Unlike Lillihook, this function does not produce a rigorously calculated 
    magnetization structure.

    Args: 
        dim: Int. Dimension of lattice. 
        chirality: String. 
            'io': inner to outer.
            'oi': outer to inner.  
        pad: Bool. Whether or not to leave some space between the edge of the 
            magnetization and the edge of the image. 
        ir: Float. Inner radius of the vortex in pixels. 
        show: Bool: If True, will show the x, y, z components in plot form. 

    Returns: (mag_x, mag_y, mag_z)
        mag_x: 2D array (dim, dim). x-component of magnetization vector. 
        mag_y: 2D array (dim, dim). y-component of magnetization vector. 
        mag_z: 2D array (dim, dim). z-component of magnetization vector. 
    """
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