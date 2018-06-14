import numpy as np
from stl import mesh
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from math import cos, sin, sqrt, acos, asin, atan, pi

from argparse import ArgumentParser


parser = ArgumentParser()
parser.add_argument("-s", "--solar", dest="solar", help="Generated Solar Object (Default is Earth)", default="Earth")
parser.add_argument("-d", "--dia", type=int, dest="dia", help="Outer Diameter Of the Planet", default="120")
parser.add_argument("-w", "--wall", type=int, dest="wall", help="Wall Thickness", default="3")
parser.add_argument("-i", "--inside", type=float, dest="inside", help="Inside Depth", default="0")
parser.add_argument("-o", "--outside", type=float, dest="outside", help="Outside Depth", default="2.2")
parser.add_argument("-f", "--freq", type=int, dest="freq", help="Division Factor", default="200")
args = parser.parse_args()

# pip install numpy-stl
# pip install tqdm
#
#####################################################################
# 2018-03-01 by Uwe Zimmermann
# uwe.zimmermann@sciencetronics.com
#####################################################################

#####################################################################
# user settings
#####################################################################
dia          = args.dia     # outer diameter of the globe
wall         = args.wall       # undisturbed wall thickness

#    the image map is carved from the outside and inside
#    into the undisturbed wall thickness
inner_depth  = args.inside     # depth of carving on inside
outer_depth  = args.outside     # depth of carving on outside

# freq=1 gives 10 pixels along the equator,
# multiplies with the division factor
freq         = args.freq      # division factor


if args.inside > 0:
	POS = "Inside"
	outer_depth  = 0     # depth of carving on outside
else:
	POS = "Outside"
	inner_depth  = 0     # depth of carving on inside

# using the grayscale map from https://asterweb.jpl.nasa.gov/gdem.asp
# mapfilename  = 'checkerboard.jpg'
# mapfilename  = 'GDEM-10km-BW.png'
mapfilename  = args.solar + "/" + args.solar + "-BW.png"

savefilename = args.solar + "/" + args.solar + "_" + POS + "_" + str(args.freq) + ".stl"

progress     = True    # display a progress bar - doesn't work in IDLE

####################################################################
# internals
####################################################################

outer_radius = dia / 2
inner_radius = outer_radius - wall

print('loading image...')
img=Image.open(mapfilename)

# downscaling the image map
longsteps = 10*freq
latsteps  = 3*freq + 1
print('resizing image...')
img=img.convert(mode="L")
img=img.resize((longsteps,latsteps),resample=Image.BICUBIC)
print('transforming image...')
pix = np.array(img.getdata()).reshape(img.size[0], img.size[1], 1, order='F')
inner_delta = inner_depth/np.amax(pix)
outer_delta = outer_depth/np.amax(pix)

def long(vector, longsteps):
    if (vector[0] > 0):
        long = longsteps/2 + atan(vector[1]/vector[0])*longsteps/(2*pi)
    elif (vector[0] < 0):
        long = longsteps + atan(vector[1]/vector[0])*longsteps/(2*pi)
    else:
        long = longsteps/2
    return int(long) % longsteps

def lat(vector, latsteps):
    longrad = sqrt(vector[0]*vector[0] + vector[1]*vector[1])
    if (longrad != 0):
        lat = latsteps/2 - atan(vector[2]/longrad)*latsteps/pi
    elif (vector[2] > 0):
        lat = 0
    else:
        lat = latsteps-1
    return int(lat)

def vnormalize(vector):
    result = np.empty([3])
    length = sqrt(np.dot(vector,vector))
    if length > 0:
        result[0] = vector[0]/length
        result[1] = vector[1]/length
        result[2] = vector[2]/length
    return result    

def divide_triangle(corner_1, corner_2, corner_3, i, j, freq):
    # linear interpolation on unit sphere using the triangular sine law
    vertex = np.empty([1, 3])
    corner_1 = vnormalize(corner_1)
    corner_2 = vnormalize(corner_2)
    corner_3 = vnormalize(corner_3)
    alpha_v1 = acos(np.dot(corner_1,corner_2)) # right side angle
    beta_v1  = i*alpha_v1/(freq)               # fraction of the angle
    alpha_v2 = acos(np.dot(corner_1,corner_3)) # left side angle
    beta_v2  = i*alpha_v2/(freq)               # fraction of the angle
    v1       = vnormalize(corner_1+vnormalize(corner_2-corner_1)*sin(beta_v1)/(sin(pi/2+alpha_v1/2-beta_v1)))
    v2       = vnormalize(corner_1+vnormalize(corner_3-corner_1)*sin(beta_v2)/(sin(pi/2+alpha_v2/2-beta_v2)))       
    alpha_v3 = acos(np.dot(v1, v2))                              
    beta_v3  = j*alpha_v3/i
    vertex   = vnormalize(v1+vnormalize(v2-v1)*sin(beta_v3)/(sin(pi/2+alpha_v3/2-beta_v3))) # get vertex
    return vertex

def geodesic(freq):
    corners = np.empty([12, 3])
    vertices = np.zeros([2, 3*freq+1, 5*freq, 3]) # inner/outer, i, k, xyz
    corners[0] = [0, 0, 1]
    a = 4/sqrt(2*(5+sqrt(5)))
    zc = 1-a*a/2
    r = sqrt(1 - zc*zc)
    i = 0
    while (i < 5):
        corners[i+1] = [r*cos(i*2*pi/5),       r*sin(i*2*pi/5),        zc]
        corners[i+6] = [r*cos((i-0.5)*2*pi/5), r*sin((i-0.5)*2*pi/5), -zc]
        i = i+1
    corners[11] = [0, 0, -1]
    i = 0
    vertices[0, 0, 0] = corners[0]
  
    # main loop calculates northern (z>0) "hemi"sphere
    if progress:
        thisrange = tqdm(range(1, freq+1))
    else:
        thisrange = range(1, freq+1)
    for i in thisrange:
        j = 0
        while (j < i):
            k = 0                                         
            while (k < 5):
                vertices[0, i, j+k*i]=vnormalize(divide_triangle(corners[0],
                                                                 corners[(k % 5) +1], 
                                                                 corners[((k+1) % 5) +1], 
                                                                 i, j, freq))
                k = k+1
            j = j+1
                               
    # going deeper into the different polyhedra                           
    # case (5): icosahedron
    # for the icosahedron it's a little bit more...
    # first the equatorial row of 2x5 triangles

    if progress:
        thisrange = tqdm(range(1, freq))
    else:
        thisrange = range(1, freq+1)
    for i in thisrange:
        j = 0        # southward opened
        while (j < i):
            k = 0
            while (k < 5):
                vertices[0, i+freq, j+k*freq]=vnormalize(divide_triangle(corners[k+1],
                                                                         corners[(k % 5) +6], 
                                                                         corners[((k+1) % 5) +6], 
                                                                         i, j, freq))
                k = k+1
            j = j+1
        j = 0      # northward opened
        while (j <= (freq-1-i)):
            k=0
            while (k < 5):
                vertices[0, i+freq, j+k*freq+i] = vnormalize(divide_triangle(corners[((k+1) % 5) +6], 
                                                                             corners[(k % 5) +1], 
                                                                             corners[((k+1) % 5) +1], 
                                                                             (freq-i), j, freq))
                k = k+1
            j = j+1
                                                
    # now apply the symmetry
    i = 3*freq
    j = 0
    vertices[0, i, j] = vertices[0, 0, 0]*[1, 1, -1]   # south pole
    # southern hemisphere
    rotsin = sin(-pi/5)
    rotcos = cos(-pi/5)

    if progress:
        thisrange = tqdm(range(1, freq+1))
    else:
        thisrange = range(1, freq+1)
    for i in thisrange:
        j = 0;
        while (j < 5*i):    # just mirror the vertices
            vertices[0, 3*freq-i, j] = [vertices[0, i, j, 0]*rotcos - vertices[0, i, j, 1]*rotsin,
                                        vertices[0, i, j, 0]*rotsin + vertices[0, i, j, 1]*rotcos,
                                        -vertices[0, i, j, 2]]
            j = j+1

    # now apply scaling and construct the inner sphere
    print('outer radius = {:1} mm, inner radius = {:1} mm'.format(outer_radius, inner_radius))

    if progress:
        thisrange = tqdm(range(3*freq+1))
    else:
        thisrange = range(3*freq+1)
    for i in thisrange:
        for k in range(5*freq):
            vector = vertices[0, i, k].copy()
            pixel  = pix[long(vector,longsteps),lat(vector,latsteps),0]
            vertices[0, i, k] = vector * (outer_radius - outer_delta*pixel) 
            vertices[1, i, k] = vector * (inner_radius + inner_delta*pixel)
    return vertices

def geodesic_mesh(freq):
  # now we should have all vertices in the array.... we JUST have to sort out, which one is where
  obj = mesh.Mesh(np.zeros(20*freq*freq*2, dtype=mesh.Mesh.dtype))
  vertices = geodesic(freq)
  n = 0

  if progress:
      thisrange = tqdm(range(0, freq))
  else:
      thisrange = range(0, freq)
  for i in thisrange:
      j = 0
      while ((j < i) or ((j == 0) and (i == 0))):
          k = 0
          while (k < 5):
              # the northern "hemi"sphere
              obj.vectors[n] = [vertices[0, i, k*i+j],
                                vertices[0, i+1, k*(i+1)+j],
                                vertices[0, i+1, (k*(i+1)+j+1) % (5*(i+1))]]
              n = n+1

              obj.vectors[n] = [vertices[1, i+1, k*(i+1)+j],
                                vertices[1, i, k*i+j],
                                vertices[1, i+1, (k*(i+1)+j+1) % (5*(i+1))]]
              n = n+1

              if (i > 0):
                  obj.vectors[n] = [vertices[0, i, k*i+j],
                                    vertices[0, i+1, (k*(i+1)+j+1) % ((i+1)*5)],
                                    vertices[0, i, (k*i+j+1) % (5*i)]]
                  n = n+1

                  obj.vectors[n] = [vertices[1, i+1, (k*(i+1)+j+1) % ((i+1)*5)],
                                    vertices[1, i, k*i+j],
                                    vertices[1, i, (k*i+j+1) % (5*i)]]
                  n = n+1
                  
              if ((j==0) and (i>0)):
                  obj.vectors[n] = [vertices[0, i, k*i+j],
                                    vertices[0, i+1, ((k+5-1) % 5)*(i+1) +i],
                                    vertices[0, i+1, k*(i+1)+j]]
                  n = n+1

                  obj.vectors[n] = [vertices[1, i+1, ((k+5-1) % 5)*(i+1) +i],
                                    vertices[1, i, k*i+j],
                                    vertices[1, i+1, k*(i+1)+j]]
                  n = n+1

            
              # southern "hemi"sphere is symmetric    
        
              obj.vectors[n] = [vertices[0, (5-2)*freq-(i+1), k*(i+1)+j],
                                vertices[0, (5-2)*freq-i, k*i+j],
                                vertices[0, (5-2)*freq-(i+1), (k*(i+1)+j+1) % ((i+1)*5)]]
              n = n+1

              obj.vectors[n] = [vertices[1, (5-2)*freq-i, k*i+j],
                                vertices[1, (5-2)*freq-(i+1), k*(i+1)+j],
                                vertices[1, (5-2)*freq-(i+1), (k*(i+1)+j+1) % ((i+1)*5)]]
              n = n+1

              if (i>0):
                  obj.vectors[n] = [vertices[0, (5-2)*freq-(i+1), (k*(i+1)+j+1) % ((i+1)*5)],
                                    vertices[0, (5-2)*freq-i, k*i+j],
                                    vertices[0, (5-2)*freq-i, (k*i+j+1) % (i*5)]]
                  n = n+1

                  obj.vectors[n] = [vertices[1, (5-2)*freq-i, k*i+j],
                                    vertices[1, (5-2)*freq-(i+1), (k*(i+1)+j+1) % ((i+1)*5)],
                                    vertices[1, (5-2)*freq-i, (k*i+j+1) % (i*5)]]
                  n = n+1

              if ((j==0) and (i>0)):
                  obj.vectors[n] = [vertices[0, (5-2)*freq-(i+1), ((k+5-1) % 5)*(i+1)+i],
                                    vertices[0, (5-2)*freq-i, k*i+j],
                                    vertices[0, (5-2)*freq-(i+1), k*(i+1)+j]]
                  n = n+1

                  obj.vectors[n] = [vertices[1, (5-2)*freq-i, k*i+j],
                                    vertices[1, (5-2)*freq-(i+1), ((k+5-1) % 5)*(i+1)+i],
                                    vertices[1, (5-2)*freq-(i+1), k*(i+1)+j]]
                  n = n+1

              k = k+1
          j = j+1
       
  # now for the equatorial faces      
  # case (5) // the icosahedron  

  if progress:
      thisrange = tqdm(range(0, freq))
  else:
      thisrange = range(0, freq)
  for i in thisrange:
      j=0;
      while (j < freq*5):
          
          obj.vectors[n] = [vertices[0, freq+i, j],
                            vertices[0, freq+i+1, j],
                            vertices[0, freq+i+1, (j+1) % (5*freq)]]
          n = n+1

          obj.vectors[n] = [vertices[1, freq+i+1, j],
                            vertices[1, freq+i, j],
                            vertices[1, freq+i+1, (j+1) % (5*freq)]]
          n = n+1

          obj.vectors[n] = [vertices[0, freq+i, j],
                            vertices[0, freq+i+1, (j+1) % (5*freq)],
                            vertices[0, freq+i, (j+1) % (5*freq)]]
          n = n+1

          obj.vectors[n] = [vertices[1, freq+i+1, (j+1) % (5*freq)],
                            vertices[1, freq+i, j],
                            vertices[1, freq+i, (j+1) % (5*freq)]]
          n = n+1
          
          j = j+1
  return obj

print('preparing geodesic...')

c = geodesic_mesh(freq)

print('writing {:}'.format(savefilename))
c.save(savefilename)

print("Ready.")
