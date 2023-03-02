import open3d
import numpy as np
from scipy.interpolate import griddata
from scipy.special import sph_harm
import glob
from PIL import Image  
import sys



def return_projection(points,normals,precision):
        points  = points - points.mean(0)
        ##project points to spherical coordinate from center
        xy    = points[:,0]**2 + points[:,1]**2
        rho   = np.sqrt(xy + points[:,2]**2)
        phi   = np.arctan2(np.sqrt(xy), points[:,2]) 

        theta = np.arctan2(points[:,1], points[:,0])


        if(phi.min()>0):
                indices_x = (phi*180*precision/np.pi).astype(np.uint16)
        else:
                indices_x = ( ((phi*180/np.pi)+90 )*precision ).astype(np.uint16)
        
        indices_y = ( ((theta*180/np.pi)+180)*precision ).astype(np.uint16)
        indices1 = np.concatenate(  [np.expand_dims(indices_x,1),np.expand_dims(indices_y,1)],1)


        sphere_colors  = np.zeros((180*precision,360*precision))
        normals_colors  = np.zeros((180*precision,360*precision,3))
        thetas  = np.zeros((180*precision,360*precision))


        sphere_colors[indices_x,indices_y] = rho
        normals_colors[indices_x,indices_y] = normals
        thetas[indices_x,indices_y] = theta
        return sphere_colors,rho,phi,thetas,normals_colors


def deproject(image,precision):
        image = image.T.reshape((image.shape[0]*image.shape[1]))

        phi = np.linspace(0,180*precision,180*precision)*np.pi/(180*precision)
        theta = (np.linspace(0,360*precision,360*precision)-180*precision)*np.pi/(180*precision)
        phi, theta = np.meshgrid(phi, theta)

        x =  np.sin(phi) * np.cos(theta)
        y =  np.sin(phi) * np.sin(theta)
        z =  np.cos(phi)

        x = np.expand_dims(np.reshape(x,(x.shape[0]*x.shape[1])),1)
        y = np.expand_dims(np.reshape(y,(y.shape[0]*y.shape[1])),1)
        z = np.expand_dims(np.reshape(z,(z.shape[0]*z.shape[1])),1)
        points = np.concatenate([x,y,z],1)
        
        points = points[image!=0.0]
        image = image[image!=0.0]

        points = points *  np.expand_dims(image,1)
        pcd = open3d.geometry.PointCloud()
        pcd.points = open3d.utility.Vector3dVector(points)
        
        # pcd.estimate_normals(search_param=open3d.geometry.KDTreeSearchParamHybrid(radius=1.0, max_nn=256))
        
        return pcd

def compute_harm(l,m):
        phi = np.linspace(0, np.pi, 1800)
        theta = np.linspace(0, 2*np.pi, 3600)
        phi, theta = np.meshgrid(phi, theta)
 

        # Calculate the spherical harmonic Y(l,m) and normalize to [0,1]
        fcolors = sph_harm(m, l, theta, phi).real
        if(l>0):
                fmax, fmin = fcolors.max(), fcolors.min()
                fcolors = (fcolors - fmin)/(fmax - fmin)
        else:
                fcolors[:] = 1.0
        fcolors = np.reshape(fcolors,(fcolors.shape[0],fcolors.shape[1])).T
        return fcolors

def fix_holes(map_to_fill):
        values_to_interpolate = np.where(map_to_fill==0.0)
        values_indices  = np.where(map_to_fill!=0.0)

        nx, ny = map_to_fill.shape[1], map_to_fill.shape[0]
        X, Y = np.meshgrid(np.arange(0, nx, 1), np.arange(0, ny, 1))


        grid_z2 = griddata((values_indices[1],values_indices[0]), map_to_fill[values_indices[0],values_indices[1]], (X, Y), method='cubic')
        grid_z2[np.isnan(grid_z2)] = 0.0

        return grid_z2

if __name__ == '__main__':
        print(sys.argv[1]) # "../Scans_01/Poisson_8/*" for example
        meshes = sorted(glob.glob(sys.argv[1]))
        for m in meshes:
                print(m)

                mesh = open3d.io.read_triangle_mesh(m)
                mesh.compute_vertex_normals()

                pcd = mesh.sample_points_uniformly(number_of_points=1000000)

                points  = np.array(pcd.points)
                normals = np.array(pcd.normals)
             
                sphere_colors,rho,phi,theta,normals_colors = return_projection(points,normals,10)
                
                grid_x  = fix_holes(normals_colors[450:1400,0])
                grid_y  = fix_holes(normals_colors[450:1400,1])
                grid_z  = fix_holes(normals_colors[450:1400,2])

                grid_x = (grid_x- grid_x.min())/(grid_x.max()-grid_x.min())
                grid_y = (grid_y- grid_y.min())/(grid_y.max()-grid_y.min())
                grid_z = (grid_z- grid_z.min())/(grid_z.max()-grid_z.min())


                image_x = Image.fromarray(grid_x.astype(np.uint8)*255).convert("L")
                image_y = Image.fromarray(grid_y.astype(np.uint8)*255).convert("L")
                image_z = Image.fromarray(grid_z.astype(np.uint8)*255).convert("L")
                
                image_x.save("./projections_x/"+m[len(sys.argv[1][:-1]):-3]+"npy")
                image_y.save("./projections_y/"+m[len(sys.argv[1][:-1]):-3]+"npy")
                image_z.save("./projections_z/"+m[len(sys.argv[1][:-1]):-3]+"npy")
