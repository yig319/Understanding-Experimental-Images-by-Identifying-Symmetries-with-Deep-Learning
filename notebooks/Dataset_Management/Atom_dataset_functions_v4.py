#!/usr/bin/env python
# coding: utf-8

import numpy as np
import math, random
from PIL.ImageDraw import Image, Draw
from shapely.geometry import Polygon, Point
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.stats import truncnorm
from sympy import symbols, Eq, solve
import time
import cv2
from scipy.ndimage import rotate # anti-clockwise

from multiprocessing import Pool
import os, re, time, shutil


def re_center_ts(ts, VA, VB, out_size):
    
    ts = np.array(ts)
    VA = np.array(VA)
    VB = np.array(VB)
    
    rep_times = 0
    new_dist, old_dist = 1, 0
    center = np.array(out_size)[:2]//3
    
    while rep_times < 4:
        rep_times = 0
        old_dist = np.sqrt( (ts[1] - center[1])**2 + (ts[0] - center[0])**2 )
        new_dist = np.sqrt( ((ts+VA)[1] - center[1])**2 + ((ts+VA)[0] - center[0])**2 )
        if new_dist < old_dist: 
            ts += VA
        else: rep_times += 1
            
        old_dist = np.sqrt( (ts[1] - center[1])**2 + (ts[0] - center[0])**2 )
        new_dist = np.sqrt( ((ts-VA)[1] - center[1])**2 + ((ts-VA)[0] - center[0])**2 )
        if new_dist < old_dist: 
            ts -= VA
        else: rep_times += 1    
            
        old_dist = np.sqrt( (ts[1] - center[1])**2 + (ts[0] - center[0])**2 )
        new_dist = np.sqrt( ((ts+VB)[1] - center[1])**2 + ((ts+VB)[0] - center[0])**2 )
        if new_dist < old_dist: 
            ts += VB
        else: rep_times += 1       
            
        old_dist = np.sqrt( (ts[1] - center[1])**2 + (ts[0] - center[0])**2 )
        new_dist = np.sqrt( ((ts-VB)[1] - center[1])**2 + ((ts-VB)[0] - center[0])**2 )
        if new_dist < old_dist: 
            ts -= VB
        else: rep_times += 1        

    return ts


def verify_image_vector(image, ts, va, vb): 
            
    ts = ts[1], ts[0]
    va = va[1], va[0]
    vb = vb[1], vb[0]

#     plt.figure(figsize=(8,8))
    plt.imshow(image)
    
    plt.ylabel('Y-axis')
    plt.xlabel('X-axis')
    ax = plt.gca()
    
    array = np.array([[ts[0], ts[1], va[0], va[1]]])
    X, Y, U, V = zip(*array)
    ax.quiver(X, Y, U, V,color='b', angles='xy', scale_units='xy', scale=1, linewidth=0.3)
    
    array = np.array([[ts[0], ts[1], vb[0], vb[1]]])
    X, Y, U, V = zip(*array)
    ax.quiver(X, Y, U, V,color='b', angles='xy', scale_units='xy', scale=1, linewidth=0.3)

    plt.draw()
    plt.show()


# basic functions:

def draw_atom(image, coordinate, r, color='white', noise=False):
    '''
    f(x) = a*(exp(-(x-mu)^2)/2*sigma^2), 
    a is the height of curve's peak, 
    mu is the position of the center of the peak,
    and sigma controls the width of the 'bell'.

    coordinate: (x, y)
    '''
    if image.dtype == np.uint8:
        image_d = np.copy(np.float32(image/255.))
    else:
        image_d = np.copy(image)

    h, w = image_d.shape[:2]
    x, y = round(coordinate[0]), round(coordinate[1])
    i, j = np.meshgrid(np.linspace(-1, 1, 2 * r), np.linspace(-1, 1, 2 * r))
    d = np.sqrt(i * i + j * j)
    sigma, mu = 0.40, 0.0
    g = np.exp(-( (d - mu) ** 2 / (2.0 * sigma ** 2)))
    g_rgb = np.dstack([g, g, g])
    g_rgb[g_rgb < 5e-2] = 0  # clean up the edges so overlapping of two atoms are not too obvious

    color_dict = {'red': 0, 'green': 1, 'blue': 2}

    if noise:
        noise_matrix = np.random.random((2 * r, 2 * r, 3))
        g_rgb = g_rgb * noise_matrix

    if x + r > w or x - r < 0 or y + r > h or y - r < 0:
        atom_padded = np.zeros((np.max((y + r, h)) - np.min((y - r, 0)), np.max((x + r, w)) - np.min((x - r, 0)), 3))
        x_ = x - np.min((0, x - r))
        y_ = y - np.min((0, y - r))

        atom_padded[int(y_ - r): int(y_ + r), x_ - r:x_ + r, :] += g_rgb
            
        atom = np.copy(atom_padded[0 - np.min((0, y - r)):h - np.min((0, y - r)), 0 - np.min((0, x - r)):w - np.min((0, x - r)), :])

    else:
        atom = np.zeros((h, w, 3))
        atom[int(y - r): int(y + r), x - r:x + r, :] += g_rgb

    image_d += atom
    return np.uint8((image_d*255))




# def draw_atom(image, coordinate, r, color='white', noise=False): # making white dot, assuming black image background
#     '''
#     f(x) = a*(exp(-(x-mu)^2)/2*sigma^2), 
#     a is the height of curve's peak, 
#     mu is the position of the center of the peak,
#     and sigma controls the width of the 'bell'.
#     '''
#     image_d = np.copy(image)
    
#     h,w = image_d.shape[:2]
#     x, y = round(coordinate[0]), round(coordinate[1])
#     i, j = np.meshgrid(np.linspace(-1,1,2*r), np.linspace(-1,1,2*r))
#     d = np.sqrt(i*i+j*j)
#     sigma, mu = 0.38, 0.0
#     g = np.exp(-( (d-mu)**2 / ( 2.0 * sigma**2 ) ) )
#     g_rgb = np.dstack([g,g,g])
#     g_rgb[g_rgb<5e-2]=0 #clean up the edges so overlapping of two atoms are not too obvious

#     color_dict = {'red':0, 'green':1, 'blue':2}

#     if x+r>w or x-r<0 or y+r>h or y-r<0:
#         atom_padded = np.zeros((np.max((y+r,h))-np.min((y-r,0)), np.max((x+r,w))-np.min((x-r,0)), 3))
#         x_ = x - np.min((0,x-r))
#         y_ = y - np.min((0,y-r))

#         if color == 'white':
#             atom_padded[int(y_-r): int(y_+r), x_-r:x_+r, :] += g_rgb
            
#         elif color == 'c1':
#             atom_padded[int(y_-r): int(y_+r), x_-r:x_+r, 0] += g
#             atom_padded[int(y_-r): int(y_+r), x_-r:x_+r, 1] += g*0.8
#             atom_padded[int(y_-r): int(y_+r), x_-r:x_+r, 2] += g*0.4
            
#         elif color == 'c2':
#             atom_padded[int(y_-r): int(y_+r), x_-r:x_+r, 0] += g
#             atom_padded[int(y_-r): int(y_+r), x_-r:x_+r, 1] += g*0.8
#             atom_padded[int(y_-r): int(y_+r), x_-r:x_+r, 2] += g*0.4
            
#         else:
#             atom_padded[int(y_-r): int(y_+r), x_-r:x_+r, color_dict[color]] += g
            
#         atom = np.copy(atom_padded[0-np.min((0,y-r)):h-np.min((0,y-r)), 0-np.min((0,x-r)):w-np.min((0,x-r)), :])

#     else:
#         atom = np.zeros((h,w,3))

#         if color == 'white':
#             atom[int(y-r): int(y+r), x-r:x+r, :] += g_rgb
            
#         elif color == 'c1':
#             atom[int(y-r): int(y+r), x-r:x+r, 0] += g
#             atom[int(y-r): int(y+r), x-r:x+r, 1] += g*0.8
#             atom[int(y-r): int(y+r), x-r:x+r, 2] += g*0.3
            
#         elif color == 'c2':
#             atom[int(y-r): int(y+r), x-r:x+r, 0] += g*0.5
#             atom[int(y-r): int(y+r), x-r:x+r, 1] += g*0.3
#             atom[int(y-r): int(y+r), x-r:x+r, 2] += g*0.8
            
#         else:
#             atom[int(y-r): int(y+r), x-r:x+r, color_dict[color]] += g
#     image_d += atom
#     return image_d


def normal_random(low, upp, num=1):
    mean = int((upp-low)/1.5)
    sd = int(1/4*(upp-low))
    return truncnorm(
        (low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd).rvs(num).astype(int)

def define_size(xa, ya, xb, yb, radius, shape, angle=None):
    x_lo_limit, y_lo_limit = np.abs(xb-xa)+radius, np.abs(yb-ya)+radius # large enough in case overlapping
    area = 8*radius**2 # by experience
    x_hi_limit = round(area/y_lo_limit)

    # default: rectangular, oblique
    unit_w = random.randint(np.min((x_lo_limit, x_hi_limit)), np.max((x_lo_limit, x_hi_limit)))
    unit_h = area/unit_w

    while np.max((unit_w, unit_h)) / np.min((unit_w, unit_h)) >1.5:
        unit_w = random.randint(np.min((x_lo_limit, x_hi_limit)), np.max((x_lo_limit, x_hi_limit)))
        unit_h = area/unit_w
    
    if shape == 'square':
        unit_w = math.sqrt(area)
        while unit_w < np.min((x_lo_limit, x_hi_limit)) and unit_w > np.max((x_lo_limit, x_hi_limit)):
            unit_w = random.randint(np.min((x_lo_limit, x_hi_limit)), np.max((x_lo_limit, x_hi_limit)))
        unit_h = area/unit_w
        
    if shape == 'hexagonal':
        unit_w = math.sqrt( area/math.sin(math.radians(60)) )
        unit_h = area/unit_w

    if shape == 'rhombic':
        (unit_h, unit_w) = (math.sqrt(2)*unit_h, math.sqrt(2)*unit_w) # maintain the same atom/area ratio

    if shape == 'triangle':
        (unit_h, unit_w) = (math.sqrt(2)*unit_h, math.sqrt(2)*unit_w) # maintain the same atom/area ratio
        unit_h, unit_w = np.sort((unit_w, unit_h)) # make sure unit_w is larger than unit_h: for better visualization
    
    if shape == 'triangle':
        unit_h = math.sqrt(area/math.tan( math.radians(angle/2) ) )
        unit_w = 2*area/unit_h

    if shape == 'triangle-right':
        unit_h = math.sqrt(2*area/math.tan(math.radians(angle)))
        unit_w = 2*area/unit_h
    return int(round(unit_w)), int(round(unit_h))

def two_point(img, radius, shape, gap=0): 
    h, w = img.shape[:2]
    
    x_low_limit, x_high_limit = 4*radius, w-4*radius
    y_low_limit, y_high_limit = 4*radius, h-4*radius

    if  x_low_limit > x_high_limit or y_low_limit > y_high_limit: 
        raise ValueError('Atom radius too large, no space for two atoms!')
    
    xa,ya,xb,yb=0,0,0,0
    vertices = [(radius,radius), (radius,h-radius), (w-radius,h-radius), (w-radius,radius)]
    poly = Polygon(vertices)
    while not poly.contains(Point(xa, ya) or not poly.contains(Point(xb, yb))):
        xa, ya = random.randint(x_low_limit, x_high_limit), random.randint(y_low_limit, y_high_limit)
        xb, yb = xa+2*radius+random.randint(0,int(gap*radius)), ya
    center = (int((xa+xb)/2), ya)  

    if shape in ['rectangular', 'square']:
        while (np.abs(xa-xb)<radius or np.abs(ya-yb)<radius): 
            angle_d = random.randint(0,359)
            xa, ya = rotate_xy(center, (xa, ya), angle_d)
            xb, yb = rotate_xy(center, (xb, yb), angle_d)

    if shape in ['oblique', 'hexagonal', 'rhombic', 'triangle', 'triangle-120', 
                 'triangle-right', 'triangle-right-45', 'triangle-right-60']:
        while np.abs(ya-yb)<radius: 
            angle_d = random.randint(0,359)
            xa, ya = rotate_xy(center, (xa, ya), angle_d)
            xb, yb = rotate_xy(center, (xb, yb), angle_d)

    return (xa, ya), (xb, yb)



def unit_cell(img, radius, shape, symmetry, boundary=False, avoid_break_symmetry=True):
    def point_to_line(p1, p2, p3):
        ''' calculate the distance from p3 perpendicular to p1,p2 line'''
        return np.linalg.norm(np.cross(p2-p1, p1-p3))/np.linalg.norm(p2-p1)   
    
    if shape == 'rectangular' or shape == 'square':
        
        radius_original = radius # radius_ is the orignal radius; radius is the one used for Polygon and draw_atom
        
        # define atom a,b positions and primitive unit cell size
        xa, ya, xb, yb, unit_w, unit_h = 0, 0, 0, 0, 0, 0
        if avoid_break_symmetry:
            lw = np.max((2, 0.2*unit_w))
            lh = np.max((2, 0.2*unit_h))
            
            while (np.abs(xa+unit_w/2-xb) < lw) and (np.abs(ya+unit_h/2-yb) < lh):
                
#             (xb,yb) == (round(xa+unit_w), round(ya+unit_h/2)): 
                (xa,ya), (xb,yb) = two_point(img, radius, shape)   
                unit_w, unit_h = define_size(xa, ya, xb, yb, radius, shape)

                # make sure point_a is closer to origin point, easier to visualize
                xa-=int(xa>xb)*unit_w
                ya-=int(ya>yb)*unit_h
                                
                
        # define angle and delta_x
        delta_x = 0
        angle_d = 90
        
        # define position for atoms in primitive unit cell:
        atom_b_list = [(xb,yb)]
        atom_a_list = [(xa,ya), (xa+unit_w, ya), 
                       (xa+unit_w-delta_x, ya+unit_h), (xa-delta_x, ya+unit_h)]

        if boundary:
            x_vertex = xa + int(xb>xa)*(unit_w) - int(xb<xa)*(unit_w)
            y_vertex = ya + int(yb>ya)*unit_h - int(yb<ya)*unit_h
            delta_x = int(xb>xa)*delta_x - int(xb<xa)*delta_x

            boundary_shape = np.array([[xa,ya], [x_vertex,ya], [x_vertex-delta_x,y_vertex], [xa-delta_x,y_vertex], [xa,ya]])

    if shape in ['oblique', 'hexagonal']:
        
        radius_original = radius # radius_ is the orignal radius; radius is the one used for Polygon and draw_atom

        # define atom a,b positions and primitive unit cell size
        xa,ya,xb,yb,unit_w,unit_h = 0,0,0,0,0,0

        condition_1 = True
        condition_2 = True

        while condition_1 or condition_2: 
            (xa,ya), (xb,yb) = two_point(img, radius, shape)   
            unit_w, unit_h = define_size(xa, ya, xb, yb, radius, shape)
            
            
            if avoid_break_symmetry:
                lh = np.max((2, 0.1*unit_h))
                condition_1 = (np.abs(ya+unit_h/2-yb) < lh) or (np.abs(ya-unit_h/2-yb) < lh)
            else:
                condition_1 = False
#             print('condition_1', lh, condition_1)
                
            # defind angle and delta_x
            if shape == 'oblique':
                xb_, yb_ = (xa+np.abs(xb-xa), ya+np.abs(yb-ya))
                angle_lo_limit = round(np.max(find_angle((xa, ya), (xb_, yb_), radius))) ### assume width for unit cell always is unit_w
                angle_hi_limit = round(np.min(find_angle((xa+unit_w, ya), (xb_, yb_), radius)))+180

                angle_d = 90
                while np.abs(angle_d-90) < 5: ### avoid to be the same with rectangular
                    angle_d = random.randint(np.min((angle_lo_limit, angle_hi_limit)), np.max((angle_lo_limit, angle_hi_limit)))
                
#                 print('before antual angle_d', 180-angle_d)
                angle_d = np.max((70, angle_d))
#                 print('antual angle_d', 180-angle_d)

            if shape == 'hexagonal': 
                angle_d = -120
            delta_x = unit_h/math.tan(math.radians(angle_d))

                            
            poly = Polygon([(xa,ya), (xa+unit_w,ya), 
                            (xa+unit_w-delta_x,ya+unit_h), 
                            (xa-delta_x,ya+unit_h), (xa,ya)])  
            condition_2 = not(Point(xb,yb).within(poly))


        # define position for atoms in primitive unit cell:
        atom_b_list = [(xb, yb)]
        atom_a_list = [(xa,ya), (xa+unit_w, ya), (round(xa+unit_w-delta_x), ya+unit_h), (round(xa-delta_x), ya+unit_h)]
        
        if boundary:
            x_vertex = xa + int(xb>xa)*(unit_w) - int(xb<xa)*(unit_w)
            y_vertex = ya + int(yb>ya)*unit_h - int(yb<ya)*unit_h
            delta_x = int(xb>xa)*delta_x - int(xb<xa)*delta_x
            
            bo_list = atom_a_list
            bo_list.append((xa,ya))
            boundary_shape = np.array((bo_list))
#             print('boundary', boundary_shape)

#             boundary_shape = np.array([[xa,ya], [x_vertex,ya], [x_vertex-delta_x,y_vertex], [xa-delta_x,y_vertex], [xa,ya]])
#             print(boundary_shape)

    if shape == 'rhombic':
        
        # define atom a,b positions and primitive unit cell size
        (xa,ya), (xb,yb) = two_point(img, radius, shape)   
        unit_w, unit_h = define_size(xa, ya, xb, yb, radius, shape)   
        poly = Polygon([(xa,ya+radius), (round(xa+unit_w/2-radius), round(ya+unit_h/2)),
                    (xa, ya+unit_h-radius), (round(xa-unit_w/2+radius), round(ya+unit_h/2))])
        
        attempts = 0
        enlarge_ratio = 1  
        radius_original = radius # radius_ is the orignal radius; radius is the one used for Polygon and draw_atom
        while (not poly.contains(Point(xb, yb))): 
            (xa,ya), (xb,yb) = two_point(img, radius_original, shape)   
            unit_w, unit_h = define_size(xa, ya, xb, yb, radius_original, shape)
            
            if avoid_break_symmetry:
                while xb == xa or yb == round(ya+unit_h/2): 
                    (xa,ya), (xb,yb) = two_point(img, radius_original, shape)   
                    unit_w, unit_h = define_size(xa, ya, xb, yb, radius_original, shape)
                        
            # make sure point_a is closer to origin point, easier for following steps
            ya-=int(ya>yb)*unit_h
            poly = Polygon([(xa,ya+radius), (round(xa+unit_w/2-radius), round(ya+unit_h/2)), (xa, ya+unit_h-radius), (round(xa-unit_w/2+radius), round(ya+unit_h/2))])
            
            attempts+=1
            if attempts%100 == 0: 
                radius -= 1

        # define angle and delta_x
        delta_x = 0
        angle_d = 90
        
        # include points in two rhombics: original one and left above one
        atom_b_list = [(xb,yb)]
        atom_a_list = [(xa,ya), (round(xa+unit_w/2), round(ya+unit_h/2)), 
                       (round(xa-unit_w/2), round(ya+unit_h/2)), (xa, ya+unit_h)]
    
        boundary_shape = np.array([[xa,ya], [round(xa+unit_w/2), round(ya+unit_h/2)], 
                           [xa, ya+unit_h], [round(xa-unit_w/2), round(ya+unit_h/2)],
                           [xa,ya]])
#         boundary_shape = tuple(map(tuple, boundary_shape))    
        
    if shape == 'triangle':
        if symmetry in ['p2', 'cm', 'pgg']: angle = random.randint(60,130)
        if symmetry in ['p31m', 'p6']: angle = 120
        if symmetry in ['p3m1']: angle = 60
        

        
        radius = 0
        while radius < 5:
            radius = random.randint(8,12)

            # define atom a,b positions and primitive unit cell size
            (xa,ya), (xb,yb) = two_point(img, radius, shape)   
            unit_w, unit_h = define_size(xa, ya, xb, yb, radius, shape, angle)
                    
            poly = Polygon([(xa,ya), (xa-unit_w/2, ya+unit_h), (xa+unit_w/2, ya+unit_h)])
            p1, p2, p3 = np.array((xa, ya)), np.array((xa-unit_w/2, ya+unit_h)), np.array((xa+unit_w/2, ya+unit_h))
            p0 = np.array((xb,yb))
            dist = np.min(( point_to_line(p1,p2,p0), point_to_line(p1,p3,p0), point_to_line(p2,p3,p0) ))                          

            attempts = 0
            radius_original = radius # radius_ is the orignal radius; radius is the one used for Polygon and draw_atom
            while not (poly.contains(Point(xb, yb)) and dist>radius): 
            # third condition is prevent mirror atoms overalpping 

                (xa,ya), (xb,yb) = two_point(img, radius_original, shape)   
                unit_w, unit_h = define_size(xa, ya, xb, yb, radius_original, shape, angle)

                if avoid_break_symmetry:
                    while xb == xa or yb == round(ya+unit_h/2): 
                        (xa,ya), (xb,yb) = two_point(img, radius_original, shape)   
                        unit_w, unit_h = define_size(xa, ya, xb, yb, radius_original, shape, angle)

                # make sure point_a is closer to origin point, easier for following steps
                ya-=int(ya>yb)*unit_h
                
                poly = Polygon([(xa,ya), (xa-unit_w/2, ya+unit_h), (xa+unit_w/2, ya+unit_h)])
                p1, p2, p3 = np.array((xa, ya)), np.array((xa-unit_w/2, ya+unit_h)), np.array((xa+unit_w/2, ya+unit_h))
                p0 = np.array((xb,yb))
                dist = np.min(( point_to_line(p1,p2,p0), point_to_line(p1,p3,p0), point_to_line(p2,p3,p0) )) 
                
                attempts+=1
                if attempts%500 == 0: radius -= 1
                    
            # define angle and delta_x
            delta_x = 0
            angle_d = 90

            # include points in two rhombics: original one and left above one
            atom_b_list = [(xb,yb)]
            atom_a_list = [(xa,ya), (round(xa+unit_w/2), ya+unit_h), 
                           (round(xa-unit_w/2), ya+unit_h), (xa,ya+unit_h)]

            boundary_shape = np.array([[xa,ya], [round(xa+unit_w/2), ya+unit_h], 
                                           [round(xa-unit_w/2), ya+unit_h], [xa,ya]])
#             boundary_shape = tuple(map(tuple, boundary_shape))

    if shape == 'triangle-right':
        if symmetry in ['cmm']: angle = random.randint(40,70)
        if symmetry in ['p6m']: angle = 60
        if symmetry in ['p4m', 'p4g']: angle = 45   
        
        radius = 0
        while radius < 5:
            radius = random.randint(8,12)
            # define atom a,b positions and primitive unit cell size
            (xa,ya), (xb,yb) = two_point(img, radius, shape) 
            unit_w, unit_h = define_size(xa, ya, xb, yb, radius, shape, angle)
            ya-=int(ya>yb)*unit_h # make sure point_a is closer to origin point, easier for following steps

            poly = Polygon([(xa, ya), (xa, ya+unit_h), (xa-unit_w, ya+unit_h)])
            p1, p2, p3 = np.array((xa, ya)), np.array((xa, ya+unit_h)), np.array((xa-unit_w, ya+unit_h))
            p0 = np.array((xb,yb))
            dist = np.min(( point_to_line(p1,p2,p0), point_to_line(p1,p3,p0), point_to_line(p2,p3,p0) ))
#             dist = np.min(( point_to_line(p1,p2,p0), point_to_line(p1,p3,p0), point_to_line(p2,p3,p0), 
#                           Point(xb,yb).distance(Point((xa-unit_w/2), ya+unit_h/2))-radius ))

            attempts = 0
            radius_original = radius # radius_ is the orignal radius; radius is the one used for Polygon and draw_atom

            while not (poly.contains(Point(xb, yb)) and dist>radius): 
                ## 0.7 is a arbitrary ratio to keep distancing and make atom position more random
            # third condition is prevent mirror atoms overalpping 

                (xa,ya), (xb,yb) = two_point(img, radius_original, shape) 
                unit_w, unit_h = define_size(xa, ya, xb, yb, radius_original, shape, angle)
                ya-=int(ya>yb)*unit_h # make sure point_a is closer to origin point, easier for following steps

                poly = Polygon([(xa, ya), (xa, ya+unit_h), (xa-unit_w, ya+unit_h)])
                p1, p2, p3 = np.array((xa, ya)), np.array((xa, ya+unit_h)), np.array((xa-unit_w, ya+unit_h))
                p0 = np.array((xb,yb))
                dist = np.min(( point_to_line(p1,p2,p0), point_to_line(p1,p3,p0), point_to_line(p2,p3,p0) ))

                attempts+=1

                if attempts%500 == 0: 
                    radius -= 1
            
            # define angle and delta_x
            delta_x = 0
            angle_d = 90
            
            # include points in two rhombics: original one and left above one
            atom_b_list = [(xb,yb)]
            atom_a_list = [(xa,ya), (xa, ya+unit_h), 
                           (xa-unit_w, ya+unit_h), (round(xa-unit_w/2), round(ya+unit_h/2))]

            boundary_shape = np.array([[xa,ya], [xa, ya+unit_h], 
                                       [xa-unit_w, ya+unit_h], [xa,ya]])

    diff = int(np.round(radius * 0.2))
    radius_b, radius_a = np.sort((radius+diff, radius_original-diff))
    
    # return radius to be radius_b
    return atom_a_list, atom_b_list, unit_w, unit_h, radius_a, radius_b, angle_d, boundary_shape


def remove_repeat_point(atom_list, r):
    new_list = []
    for i, point in enumerate(atom_list):
        d = True
        for p in atom_list[i+1:]:
            if np.linalg.norm(np.array(point)-np.array(p))<r: d=False
        if d: new_list.append(point)
    return new_list



def draw_atoms_with_boundary(img, radius_a, radius_b, atom_a_all, atom_b_all, boundary_shape, visualize=False):
    # draw boundary
    img_ = np.copy(img)
    
    result_uc = np.copy(img_)
    if visualize:        
        img_ = (img_*255).astype(np.uint8)
        img_PIL = Image.fromarray(img_)
        img_PIL_draw = Draw(img_PIL) 

        img_PIL_draw.line(tuple(map(tuple, boundary_shape)), fill='white', width = 2) 
        result_uc = (np.array(img_PIL)/255).astype(np.float)
#         result_uc = draw_boundary(result_uc, boundary)

    # draw_all_atoms
    atom_a_all = remove_repeat_point(atom_a_all, radius_a)    
    atom_b_all = remove_repeat_point(atom_b_all, radius_b)
    
    for point_a in atom_a_all:
        result_uc = draw_atom(result_uc, point_a, radius_a, color='white', noise=True)
    for point_b in atom_b_all:
        result_uc = draw_atom(result_uc, point_b, radius_b, color='white', noise=True) 
    
    if visualize:
        plt.imshow(result_uc)
        plt.show()
    return result_uc


def find_angle(p1, p2, r):
    '''p1: starting point
       p2: circle center
       r: circle radius
    '''
    x0, y0 = p1
    x1, y1 = p2

    m = symbols('m')
    eq1 = Eq((m*x0-y0-m*x1+y1)**2-r**2*(m**2+1), 0)
    sol = solve(eq1)

    angle = []
    for s in sol:
        angle.append(math.degrees(math.atan(s)))
    
    if len(angle)==1:
        if x0-x1 == r: angle.append(-90)
        elif x1-x0 == r: angle.append(90)
        else: angle.append(0)
        sol.append(0)
    return angle


# transformation functions:
def mirror(position, axis_start_point, axis_end_point, origin=(0,0)):
    (x,y) = position
    vector = -( np.array((x,y)) - ( np.array(axis_start_point) - np.array(origin) ) )
    axis = np.array(axis_end_point) - np.array(axis_start_point)
    x_mir, y_mir = (vector - 2*np.dot(np.dot(vector, axis), axis)/(np.linalg.norm(axis)**2))
    x, y =  (np.array((x_mir, y_mir)) + ( np.array(axis_start_point) - np.array(origin) ) )
    return int(round(x)), int(round(y))

def glide(position, axis_start_point, axis_end_point):
    x_mir, y_mir = mirror(position, axis_start_point, axis_end_point)
    x, y = np.array((x_mir, y_mir)) + np.array(axis_end_point) - np.array(axis_start_point)
    return int(round(x)), int(round(y))

def rotate_xy(origin, point, angle):
    """
    Rotate a point counterclockwise by a given angle around a given origin.
    """
    oy, ox = origin
    py, px = point
    
    theta = np.radians(-angle)
    qx = ox + math.cos(theta) * (px - ox) - math.sin(theta) * (py - oy)
    qy = oy + math.sin(theta) * (px - ox) + math.cos(theta) * (py - oy)
    return int(qy), int(qx)


def get_unique_points_in_unit_cell(atom_list, t_va, t_vb, radius):
    t_va, t_vb = np.array((t_va[1], t_va[0])), np.array((t_vb[1], t_vb[0]))
    
    p_unique_list = []
    for i, p in enumerate(atom_list):
        d = True
        for p_translated in [p+t_va, p+t_vb, p+t_va+t_vb, p-t_va, p-t_vb, p-t_va-t_vb,
                             p+t_va-t_vb, p-t_va+t_vb]:
            for p_compare in atom_list[i+1:]:
                if np.linalg.norm(np.array(p_translated)-np.array(p_compare))<radius: 
                    d = False
                    continue
        if d: p_unique_list.append(p)
    return p_unique_list

def re_center_point(point, t_va, t_vb, out_size):
    point = np.array(point)
    t_va, t_vb = np.array((t_va[1], t_va[0])), np.array((t_vb[1], t_vb[0]))
    center = np.array((out_size[0]//2, out_size[1]//2))
    new_point = point + t_va # make new_point different 
    
    while (new_point != point).any():
        point = np.copy(new_point)
        for p in [new_point+t_va, new_point+t_vb, new_point-t_va, new_point-t_vb, 
                  new_point+t_va+t_vb, new_point+t_va-t_vb, new_point-t_va+t_vb, new_point-t_va-t_vb]:
            if np.linalg.norm(p-center) < np.linalg.norm(point-center):
                new_point = np.copy(p)
                continue
    return point


def get_all_points_by_vector_new(point, out_size, t_va, t_vb, radius):

    y, x = point
    all_points = []
    for s in range(-out_size[0], out_size[0]):
        for t in range(-out_size[1], out_size[1]):
            pos = (x + s*t_va[0] + t*t_vb[0], y + s*t_va[1] + t*t_vb[1])
            if 0 <= pos[0] < out_size[0] and 0 <= pos[1] < out_size[1]:
                all_points.append((pos[1], pos[0]))
    return all_points

# def find_corners(point, VA, VB, out_size):
#     '''
#     for best usage, start from center point
#     '''
# #     point = np.array((point[1], point[0]))
#     VA, VB = np.array((VA[1], VA[0])), np.array((VB[1], VB[0]))

#     save_point = np.copy(point)

#     center = np.array((out_size[0]//2, out_size[1]//2))
    
#     corners = []
#     point = np.copy(save_point)
#     new_point = point + VA  # make new_point different 
#     while (new_point != point).any() and new_point[0]<out_size[0] and new_point[1]<out_size[1] and new_point[0]>0 and new_point[1]>0:
#         point = np.copy(new_point)
#         for p in [new_point+VA, new_point+VB, new_point+VA+VB]:
#             if np.linalg.norm(p-center) > np.linalg.norm(point-center):
#                 new_point = np.copy(p)
#                 continue
#     corners.append(tuple(point)) # last step point in case out of range

#     point = np.copy(save_point)
#     new_point = point + VA  # make new_point different 
#     while (new_point != point).any() and new_point[0]<out_size[0] and new_point[1]<out_size[1] and new_point[0]>0 and new_point[1]>0:
#         point = np.copy(new_point)
#         for p in [new_point-VA, new_point-VB, new_point-VA-VB]:
#             if np.linalg.norm(p-center) > np.linalg.norm(point-center):
#                 new_point = np.copy(p)
#                 continue
#     corners.append(tuple(point))

#     point = np.copy(save_point)
#     new_point = point + VA  # make new_point different 
#     while (new_point != point).any() and new_point[0]<out_size[0] and new_point[1]<out_size[1] and new_point[0]>0 and new_point[1]>0:
#         point = np.copy(new_point)
#         for p in [new_point+VA, new_point-VB, new_point+VA-VB]:
#             if np.linalg.norm(p-center) > np.linalg.norm(point-center):
#                 new_point = np.copy(p)
#                 continue
#     corners.append(tuple(point))

#     point = np.copy(save_point)
#     new_point = point + VA  # make new_point different 
#     while (new_point != point).any() and new_point[0]<out_size[0] and new_point[1]<out_size[1] and new_point[0]>0 and new_point[1]>0:
#         point = np.copy(new_point)
#         for p in [new_point+VB, new_point-VA, new_point-VA+VB]:
#             if np.linalg.norm(p-center) > np.linalg.norm(point-center) and p[0]<out_size[0] and p[1]<out_size[1] and p[0]>0 and p[0]>0:
#                 new_point = np.copy(p)
#                 continue
#     corners.append(tuple(point))
#     return corners


# def get_all_points_by_vector_new(point, out_size, t_va, t_vb, radius):      

#     point = re_center_point(point, t_va, t_vb, out_size) 
# #     corners = find_corners(point, t_va, t_vb, out_size)  
#     corners = [point]
# #     print(corners)
    
#     t_va, t_vb = np.array((t_va[1], t_va[0])), np.array((t_vb[1], t_vb[0]))

#     a_margin = out_size[0]//10
#     b_margin = out_size[0]//10
#     all_points = []
    
#     # t_va, then t_vb
#     for point in corners:
#         p = np.copy(point)
#         t_va_saved = np.copy(p)
#         while p[1] > -a_margin and p[0] > -a_margin and p[1] < out_size[1]+a_margin and p[0] < out_size[0]+a_margin: 
#             all_points.append(tuple(p))
#             t_va_saved = np.copy(p)
#             while p[1] > -b_margin and p[0] > -b_margin and p[1] < out_size[1]+b_margin and p[0] < out_size[0]+b_margin: 
#                 p+=t_vb
#                 all_points.append(tuple(p))

#             p = np.copy(t_va_saved)
#             while p[1] > -b_margin and p[0] > -b_margin and p[1] < out_size[1]+b_margin and p[0] < out_size[0]+b_margin: 
#                 p-=t_vb
#                 all_points.append(tuple(p))

#             p = t_va_saved + t_va
            
#         p = np.copy(point)
#         t_va_saved = np.copy(p)
#         while p[1] > -a_margin and p[0] > -a_margin and p[1] < out_size[1]+a_margin and p[0] < out_size[0]+a_margin: 
#             all_points.append(tuple(p))

#             t_va_saved = np.copy(p)
#             while p[1] > -b_margin and p[0] > -b_margin and p[1] < out_size[1]+b_margin and p[0] < out_size[0]+b_margin: 
#                 p+=t_vb
#                 all_points.append(tuple(p))

#             p = np.copy(t_va_saved)
#             while p[1] > -b_margin and p[0] > -b_margin and p[1] < out_size[1]+b_margin and p[0] < out_size[0]+b_margin: 
#                 p-=t_vb
#                 all_points.append(tuple(p))

#             p = t_va_saved - t_va
            
#     a_margin = out_size[0]//10
#     b_margin = out_size[0]//10
#     # t_vb, then t_va
#     for point in corners:
#         p = np.copy(point)
#         t_vb_saved = np.copy(p)
#         while p[1] > -b_margin and p[0] > -b_margin and p[1] < out_size[1]+b_margin and p[0] < out_size[0]+b_margin: 
#             all_points.append(tuple(p))
#             t_vb_saved = np.copy(p)
#             while p[1] > -a_margin and p[0] > -a_margin and p[1] < out_size[1]+a_margin and p[0] < out_size[0]+a_margin: 
#                 p+=t_va
#                 all_points.append(tuple(p))

#             p = np.copy(t_vb_saved)
#             while p[1] > -a_margin and p[0] > -a_margin and p[1] < out_size[1]+a_margin and p[0] < out_size[0]+a_margin: 
#                 p-=t_va
#                 all_points.append(tuple(p))

#             p = t_vb_saved + t_vb

            
#         p = np.copy(point)
#         t_vb_saved = np.copy(p)
#         while p[1] > -b_margin and p[0] > -b_margin and p[1] < out_size[1]+b_margin and p[0] < out_size[0]+b_margin: 
#             all_points.append(tuple(p))

#             t_vb_saved = np.copy(p)
#             while p[1] > -a_margin and p[0] > -a_margin and p[1] < out_size[1]+a_margin and p[0] < out_size[0]+a_margin: 
#                 p+=t_vb
#                 all_points.append(tuple(p))

#             p = np.copy(t_vb_saved)
#             while p[1] > -a_margin and p[0] > -a_margin and p[1] < out_size[1]+a_margin and p[0] < out_size[0]+a_margin: 
#                 p-=t_vb
#                 all_points.append(tuple(p))

#             p = t_vb_saved - t_vb  
            
            
#     return all_points



def wp_atoms(out_size, symmetry, shape='random', rotation=True, timing=False,
             show_step=False):

    image = np.zeros(out_size)
    height, width = image.shape[:2]
    
# define parameters
    
    ## primitive unit cell parameters:
    radius_a = random.randint(8,12)

    start = time.time()
    ## translation unit cell paramters
    if symmetry == 'p1':
        if shape == 'random':
            shape = random.choice(['square', 'rectangular', 'oblique', 'rhombic', 'hexagonal'])
        
        # define parameters and atoms in unit cell
        atom_a_list, atom_b_list, unit_w, unit_h, radius_a, radius_b, angle_d, boundary_shape = unit_cell(image, radius_a, shape, symmetry, boundary=True, avoid_break_symmetry=True)        
        if shape in ['square', 'rectangular', 'oblique', 'hexagonal']:                
            repeat_w, repeat_h = unit_w, unit_h
            delta_w = round(repeat_h/math.tan(math.radians(angle_d)), 2)
        if shape == 'rhombic':
            repeat_w, repeat_h = unit_w, unit_h
            
        ## remove repeating atoms - point=(x,y)
        atom_a_unit_cell_list = remove_repeat_point(atom_a_list, radius_a) 
        atom_b_unit_cell_list = remove_repeat_point(atom_b_list, radius_b)
        
        
    if symmetry in ['pm']:
        shape = 'rectangular'
        atom_a_list, atom_b_list, unit_w, unit_h, radius_a, radius_b, angle_d, boundary_shape = unit_cell(image, radius_a, shape, symmetry, boundary=True, avoid_break_symmetry=True)        
        
        repeat_w, repeat_h = unit_w, 2*unit_h
        delta_w = round(repeat_h/math.tan(math.radians(angle_d)), 2)
        
        # default: horizontal mirror 
        atom_a_unit_cell_list, atom_b_unit_cell_list = atom_a_list.copy(), atom_b_list.copy()

        copy_a, copy_b = atom_a_unit_cell_list.copy(), atom_b_unit_cell_list.copy()
        axis_start = atom_a_unit_cell_list[0][0], atom_a_unit_cell_list[0][1]+unit_h
        axis_end = atom_a_unit_cell_list[0][0]+unit_w, atom_a_unit_cell_list[0][1]+unit_h

        for point in copy_a:
            atom_a_unit_cell_list.append(mirror(point, axis_start, axis_end))
        for point in copy_b:
            atom_b_unit_cell_list.append(mirror(point, axis_start, axis_end))
    
    if symmetry in ['pg']:
        shape = 'rectangular'
        atom_a_list, atom_b_list, unit_w, unit_h, radius_a, radius_b, angle_d, boundary_shape = unit_cell(image, radius_a, shape, symmetry, boundary=True, avoid_break_symmetry=True)        
        
        repeat_w, repeat_h = unit_w, 2*unit_h
        delta_w = round(repeat_h/math.tan(math.radians(angle_d)), 2)
        
            
        # default: vertical glide 
        atom_a_unit_cell_list, atom_b_unit_cell_list = atom_a_list.copy(), atom_b_list.copy()

        copy_a, copy_b = atom_a_unit_cell_list.copy(), atom_b_unit_cell_list.copy()
        axis_start = atom_a_unit_cell_list[0][0]+unit_w/2, atom_a_unit_cell_list[0][1]
        axis_end = atom_a_unit_cell_list[0][0]+unit_w/2, atom_a_unit_cell_list[0][1]+unit_h

        for point in copy_a:
            atom_a_unit_cell_list.append(glide(point, axis_start, axis_end))
        for point in copy_b:
            atom_b_unit_cell_list.append(glide(point, axis_start, axis_end))
    
    
    if symmetry in ['pmm']:
        shape = 'rectangular'
        atom_a_list, atom_b_list, unit_w, unit_h, radius_a, radius_b, angle_d, boundary_shape = unit_cell(image, radius_a, shape, symmetry, boundary=True, avoid_break_symmetry=True)        

        repeat_w, repeat_h = 2*unit_w, 2*unit_h
        delta_w = round(repeat_h/math.tan(math.radians(angle_d)), 2)
        

        atom_a_unit_cell_list, atom_b_unit_cell_list = atom_a_list.copy(), atom_b_list.copy()

        copy_a, copy_b = atom_a_unit_cell_list.copy(), atom_b_unit_cell_list.copy()
        
        axis_start = atom_a_unit_cell_list[0][0], atom_a_unit_cell_list[0][1]+unit_h
        axis_end = atom_a_unit_cell_list[0][0]+unit_w, atom_a_unit_cell_list[0][1]+unit_h
        for point in copy_a:
            atom_a_unit_cell_list.append(mirror(point, axis_start, axis_end))
        for point in copy_b:
            atom_b_unit_cell_list.append(mirror(point, axis_start, axis_end))
        
        copy_a, copy_b = atom_a_unit_cell_list.copy(), atom_b_unit_cell_list.copy()
        axis_start = atom_a_unit_cell_list[0][0]+unit_w, atom_a_unit_cell_list[0][1]
        axis_end = atom_a_unit_cell_list[0][0]+unit_w, atom_a_unit_cell_list[0][1]+unit_h
        for point in copy_a:
            atom_a_unit_cell_list.append(mirror(point, axis_start, axis_end))
        for point in copy_b:
            atom_b_unit_cell_list.append(mirror(point, axis_start, axis_end))    

            
    if symmetry in ['pmg']:
        shape = 'rectangular'
        atom_a_list, atom_b_list, unit_w, unit_h, radius_a, radius_b, angle_d, boundary_shape = unit_cell(image, radius_a, shape, symmetry, boundary=True, avoid_break_symmetry=True)        

        if timing: print('unit cell', time.time() - start)
        
        repeat_w, repeat_h = 2*unit_w, 2*unit_h
        delta_w = round(repeat_h/math.tan(math.radians(angle_d)), 2)
    
    
        # default: horizontal mirror 
        atom_a_unit_cell_list, atom_b_unit_cell_list = atom_a_list.copy(), atom_b_list.copy()

        copy_a, copy_b = atom_a_unit_cell_list.copy(), atom_b_unit_cell_list.copy()

        center = atom_a_unit_cell_list[0][0]+unit_w, round(atom_a_unit_cell_list[0][1]+unit_h/2)        
        for point in copy_a:
            atom_a_unit_cell_list.append(rotate_xy(center, point, 180))
        for point in copy_b:
            atom_b_unit_cell_list.append(rotate_xy(center, point, 180))

        copy_a, copy_b = atom_a_unit_cell_list.copy(), atom_b_unit_cell_list.copy()

        axis_start = atom_a_unit_cell_list[0][0], atom_a_unit_cell_list[0][1]+unit_h
        axis_end = atom_a_unit_cell_list[0][0]+unit_w, atom_a_unit_cell_list[0][1]+unit_h
        for point in copy_a:
            atom_a_unit_cell_list.append(mirror(point, axis_start, axis_end))
        for point in copy_b:
            atom_b_unit_cell_list.append(mirror(point, axis_start, axis_end))
            
        if timing: print('translation unit cell', time.time() - start)
            
  
    if symmetry in ['p4']:
        shape = 'square'
        atom_a_list, atom_b_list, unit_w, unit_h, radius_a, radius_b, angle_d, boundary_shape = unit_cell(image, radius_a, shape, symmetry, boundary=True, avoid_break_symmetry=True)        

        repeat_w, repeat_h = 2*unit_w, 2*unit_h
        delta_w = round(repeat_h/math.tan(math.radians(angle_d)), 2)
        
        
        # default: horizontal mirror 
        atom_a_unit_cell_list, atom_b_unit_cell_list = atom_a_list.copy(), atom_b_list.copy()

        copy_a, copy_b = atom_a_unit_cell_list.copy(), atom_b_unit_cell_list.copy()
        center = copy_a[0][0]+unit_w, copy_a[0][1]+unit_h        
        for point in copy_a:
            atom_a_unit_cell_list.append(rotate_xy(center, point, 90))
        for point in copy_b:
            atom_b_unit_cell_list.append(rotate_xy(center, point, 90))
        
        for point in copy_a:
            atom_a_unit_cell_list.append(rotate_xy(center, point, 180))
        for point in copy_b:
            atom_b_unit_cell_list.append(rotate_xy(center, point, 180))
        
        for point in copy_a:
            atom_a_unit_cell_list.append(rotate_xy(center, point, -90))
        for point in copy_b:
            atom_b_unit_cell_list.append(rotate_xy(center, point, -90))        
        
        
    if symmetry == 'p3':
        shape = 'hexagonal'
        atom_a_list, atom_b_list, unit_w, unit_h, radius_a, radius_b, angle_d, boundary_shape = unit_cell(image, radius_a, shape, symmetry, boundary=True, avoid_break_symmetry=True)        
        center = atom_a_list[1]
        for i, pa in enumerate(atom_a_list):
            atom_a_list[i] = rotate_xy(center, pa, 90)
        for i, pb in enumerate(atom_b_list):
            atom_b_list[i] = rotate_xy(center, pb, 90)
        for i, v in enumerate(boundary_shape):
            boundary_shape[i] = rotate_xy(center, v, 90)

        angle_d = 60 # fix the angle to 60 for translation unit cell
        repeat_w, repeat_h = 2*unit_h, round(1.5*unit_w)
        delta_w = round(repeat_h/math.tan(math.radians(angle_d)), 2)
      
        ## remove repeating atoms
        atom_a_unit_cell_list = remove_repeat_point(atom_a_list, radius_a)
        atom_b_unit_cell_list = remove_repeat_point(atom_b_list, radius_b)

            
        copy_a, copy_b = atom_a_unit_cell_list.copy(), atom_b_unit_cell_list.copy()
        center = atom_a_unit_cell_list[1]
        for point in copy_a:
            atom_a_unit_cell_list.append(rotate_xy(center, point, 120))
        for point in copy_b:
            atom_b_unit_cell_list.append(rotate_xy(center, point, 120))
        for point in copy_a:
            atom_a_unit_cell_list.append(rotate_xy(center, point, -120))
        for point in copy_b:
            atom_b_unit_cell_list.append(rotate_xy(center, point, -120))  
            
        center = atom_a_unit_cell_list[3]
        for point in copy_a:
            atom_a_unit_cell_list.append(rotate_xy(center, point, 120))
        for point in copy_b:
            atom_b_unit_cell_list.append(rotate_xy(center, point, 120))
        for point in copy_a:
            atom_a_unit_cell_list.append(rotate_xy(center, point, -120))
        for point in copy_b:
            atom_b_unit_cell_list.append(rotate_xy(center, point, -120))    

    if symmetry == 'p2':
        if shape == 'random':
            shape = random.choice(['square', 'rectangular', 'oblique', 'triangle', 'hexagonal'])
        
        # define parameters and atoms in unit cell
        atom_a_list, atom_b_list, unit_w, unit_h, radius_a, radius_b, angle_d, boundary_shape = unit_cell(image, radius_a, shape, symmetry, boundary=True, avoid_break_symmetry=True)        
        if shape in ['rectangular', 'square', 'oblique', 'hexagonal']:
            repeat_w, repeat_h = unit_w, 2*unit_h
            delta_w = round(repeat_h/math.tan(math.radians(angle_d)), 2)
        if shape == 'triangle':
            repeat_w, repeat_h = unit_w, 2*unit_h
            delta_w = round(repeat_h/math.tan(math.radians(angle_d)), 2)
            
        ## remove repeating atoms
        atom_a_unit_cell_list = remove_repeat_point(atom_a_list, radius_a)
        atom_b_unit_cell_list = remove_repeat_point(atom_b_list, radius_b)

        
        if shape == 'rectangular' or  shape == 'square':
            center = round(atom_a_unit_cell_list[0][0]+unit_w/2), atom_a_unit_cell_list[0][1]+unit_h
        if shape == 'oblique' or shape == 'hexagonal':
            l = np.array(atom_a_unit_cell_list)
            l_new = l[l[:,1]==np.min(l[:,1])]
            l_new_combine = l_new[:,0]+l_new[:,1]
            small = l_new[np.argmin(l_new_combine)]
            
            center = round(small[0]+unit_w/2-delta_w/2), small[1]+unit_h
        if shape == 'triangle':
            center = round(atom_a_unit_cell_list[0][0]), atom_a_unit_cell_list[0][1]+unit_h

        copy_a = atom_a_unit_cell_list.copy()
        copy_b = atom_b_unit_cell_list.copy()
        
        for point in copy_a:
            atom_a_unit_cell_list.append(rotate_xy(center, point, 180))
        for point in copy_b:
            atom_b_unit_cell_list.append(rotate_xy(center, point, 180))   

    
    if symmetry == 'cm':
        shape = 'triangle'
        atom_a_list, atom_b_list, unit_w, unit_h, radius_a, radius_b, angle_d, boundary_shape = unit_cell(image, radius_a, shape, symmetry, boundary=True, avoid_break_symmetry=True)        

        repeat_w, repeat_h = unit_w, 2*unit_h
        delta_w = round(repeat_h/math.tan(math.radians(angle_d)), 2)
        
        ## remove repeating atoms
        atom_a_unit_cell_list = remove_repeat_point(atom_a_list, radius_a)
        atom_b_unit_cell_list = remove_repeat_point(atom_b_list, radius_b)


        copy_a, copy_b = atom_a_unit_cell_list.copy(), atom_b_unit_cell_list.copy()
        axis_start = atom_a_unit_cell_list[0][0], atom_a_unit_cell_list[0][1]+unit_h
        axis_end = atom_a_unit_cell_list[0][0]+unit_w, atom_a_unit_cell_list[0][1]+unit_h
        for point in copy_a:
            atom_a_unit_cell_list.append(mirror(point, axis_start, axis_end))
        for point in copy_b:
            atom_b_unit_cell_list.append(mirror(point, axis_start, axis_end))
        
            
    if symmetry == 'pgg':
        shape = 'triangle'
        atom_a_list, atom_b_list, unit_w, unit_h, radius_a, radius_b, angle_d, boundary_shape = unit_cell(image, radius_a, shape, symmetry, boundary=True, avoid_break_symmetry=True)        

        repeat_w, repeat_h = unit_w, 2*unit_h
        delta_w = round(repeat_h/math.tan(math.radians(angle_d)), 2)
        
        ## remove repeating atoms
        atom_a_unit_cell_list = remove_repeat_point(atom_a_list, radius_a)
        atom_b_unit_cell_list = remove_repeat_point(atom_b_list, radius_b)

            
        copy_a, copy_b = atom_a_unit_cell_list.copy(), atom_b_unit_cell_list.copy()
        axis_start = atom_a_unit_cell_list[0][0], round(atom_a_unit_cell_list[0][1]+unit_h/2)
        axis_end = round(atom_a_unit_cell_list[0][0]+unit_w/2), round(atom_a_unit_cell_list[0][1]+unit_h/2)
        for point in copy_a:
            atom_a_unit_cell_list.append(glide(point, axis_start, axis_end))
            atom_a_unit_cell_list.append(glide(point, axis_end, axis_start))
        for point in copy_b:
            atom_b_unit_cell_list.append(glide(point, axis_start, axis_end))
            atom_b_unit_cell_list.append(glide(point, axis_end, axis_start))
            
        copy_a, copy_b = atom_a_unit_cell_list.copy(), atom_b_unit_cell_list.copy()
        center = round(atom_a_unit_cell_list[0][0]), atom_a_unit_cell_list[0][1]+unit_h
        for point in copy_a:
            atom_a_unit_cell_list.append(rotate_xy(center, point, 180))
        for point in copy_b:
            atom_b_unit_cell_list.append(rotate_xy(center, point, 180))

            
            
    if symmetry == 'p31m':
        shape = 'triangle'
        atom_a_list, atom_b_list, unit_w, unit_h, radius_a, radius_b, angle_d, boundary_shape = unit_cell(image, radius_a, shape, symmetry, boundary=True, avoid_break_symmetry=True)        
        
        angle_d = 60 # fix the angle to 60 for translation unit cell
        repeat_w, repeat_h = unit_w, 3*unit_h
        delta_w = round(repeat_h/math.tan(math.radians(angle_d)), 2)

        ## remove repeating atoms
        atom_a_unit_cell_list = remove_repeat_point(atom_a_list, radius_a)
        atom_b_unit_cell_list = remove_repeat_point(atom_b_list, radius_b)

        
        copy_a, copy_b = atom_a_unit_cell_list.copy(), atom_b_unit_cell_list.copy()
        xa,ya = atom_a_unit_cell_list[0]
        center = xa, ya
        for point in copy_a:
            atom_a_unit_cell_list.append(rotate_xy(center, point, 120))
        for point in copy_b:
            atom_b_unit_cell_list.append(rotate_xy(center, point, 120))
        
        for point in copy_a:
            atom_a_unit_cell_list.append(rotate_xy(center, point, -120))
        for point in copy_b:
            atom_b_unit_cell_list.append(rotate_xy(center, point, -120))        
        
        copy_a, copy_b = atom_a_unit_cell_list.copy(), atom_b_unit_cell_list.copy()
        axis_start = xa, round(ya-2*unit_h)
        axis_end = round(xa+unit_w/2), ya+unit_h
        for point in copy_a:
            atom_a_unit_cell_list.append(mirror(point, axis_start, axis_end))
        for point in copy_b:
            atom_b_unit_cell_list.append(mirror(point, axis_start, axis_end))
                
                
    if symmetry == 'p6':
        shape = 'triangle'
        atom_a_list, atom_b_list, unit_w, unit_h, radius_a, radius_b, angle_d, boundary_shape = unit_cell(image, radius_a, shape, symmetry, boundary=True, avoid_break_symmetry=True)        
        
        angle_d = 60 # fix the angle to 60 for translation unit cell
        repeat_w, repeat_h = unit_w, 3*unit_h
        delta_w = round(repeat_h/math.tan(math.radians(angle_d)), 2)

        ## remove repeating atoms
        atom_a_unit_cell_list = remove_repeat_point(atom_a_list, radius_a)
        atom_b_unit_cell_list = remove_repeat_point(atom_b_list, radius_b)

            
        copy_a, copy_b = atom_a_unit_cell_list.copy(), atom_b_unit_cell_list.copy()
        xa, ya = atom_a_unit_cell_list[0]
        center = xa, ya
        for point in copy_a:
            atom_a_unit_cell_list.append(rotate_xy(center, point, 120))
        for point in copy_b:
            atom_b_unit_cell_list.append(rotate_xy(center, point, 120))
        
        for point in copy_a:
            atom_a_unit_cell_list.append(rotate_xy(center, point, -120))
        for point in copy_b:
            atom_b_unit_cell_list.append(rotate_xy(center, point, -120))  
        

        copy_a, copy_b = atom_a_unit_cell_list.copy(), atom_b_unit_cell_list.copy()        
        center = (boundary_shape[1][0]-int(unit_w/4), boundary_shape[1][1]-int(unit_h*1.5))
        for point in copy_a:
            atom_a_unit_cell_list.append(rotate_xy(center, point, 180))
        for point in copy_b:
            atom_b_unit_cell_list.append(rotate_xy(center, point, 180))            
            
            
    if symmetry == 'p3m1':
        shape = 'triangle'
        atom_a_list, atom_b_list, unit_w, unit_h, radius_a, radius_b, angle_d, boundary_shape = unit_cell(image, radius_a, shape, symmetry, boundary=True, avoid_break_symmetry=True)        
        
        angle_d = 60 # fix the angle to 60 for translation unit cell
        repeat_w, repeat_h = 3*unit_w, 2*unit_h
      
        ## remove repeating atoms
        atom_a_unit_cell_list = remove_repeat_point(atom_a_list, radius_a)
        atom_b_unit_cell_list = remove_repeat_point(atom_b_list, radius_b)

            
        copy_a, copy_b = atom_a_unit_cell_list.copy(), atom_b_unit_cell_list.copy()
        xa,ya = atom_a_unit_cell_list[0]
        axis_start = round(xa+unit_w/2), ya+unit_h
        axis_end = round(xa-unit_w/2), ya+unit_h
        for point in copy_a:
            atom_a_unit_cell_list.append(mirror(point, axis_start, axis_end))
        for point in copy_b:
            atom_b_unit_cell_list.append(mirror(point, axis_start, axis_end))  
            
        copy_a, copy_b = atom_a_unit_cell_list.copy(), atom_b_unit_cell_list.copy()
        
        center = axis_start
        for point in copy_a:
            atom_a_unit_cell_list.append(rotate_xy(center, point, 120))
        for point in copy_b:
            atom_b_unit_cell_list.append(rotate_xy(center, point, 120))
        for point in copy_a:
            atom_a_unit_cell_list.append(rotate_xy(center, point, -120))
        for point in copy_b:
            atom_b_unit_cell_list.append(rotate_xy(center, point, -120))  
            
        center = axis_end
        for point in copy_a:
            atom_a_unit_cell_list.append(rotate_xy(center, point, 120))
        for point in copy_b:
            atom_b_unit_cell_list.append(rotate_xy(center, point, 120))
        for point in copy_a:
            atom_a_unit_cell_list.append(rotate_xy(center, point, -120))
        for point in copy_b:
            atom_b_unit_cell_list.append(rotate_xy(center, point, -120))                          
            
    if symmetry in ['cmm']:
        shape = 'triangle-right'
        atom_a_list, atom_b_list, unit_w, unit_h, radius_a, radius_b, angle_d, boundary_shape = unit_cell(image, radius_a, shape, symmetry, boundary=True, avoid_break_symmetry=True)        

        repeat_w, repeat_h = 2*unit_w, 2*unit_h
        
        ## remove repeating atoms
        atom_a_unit_cell_list = remove_repeat_point(atom_a_list, radius_a)
        atom_b_unit_cell_list = remove_repeat_point(atom_b_list, radius_b)

        xa,ya = atom_a_unit_cell_list[0]
        copy_a, copy_b = atom_a_unit_cell_list.copy(), atom_b_unit_cell_list.copy()        
        axis_start = xa-unit_w, ya+unit_h
        axis_end = xa+unit_w, ya+unit_h
        for point in copy_a:
            atom_a_unit_cell_list.append(mirror(point, axis_start, axis_end))
        for point in copy_b:
            atom_b_unit_cell_list.append(mirror(point, axis_start, axis_end))

        copy_a, copy_b = atom_a_unit_cell_list.copy(), atom_b_unit_cell_list.copy()
        axis_start = xa, ya
        axis_end = xa, ya+2*unit_h
        for point in copy_a:
            atom_a_unit_cell_list.append(mirror(point, axis_start, axis_end))
        for point in copy_b:
            atom_b_unit_cell_list.append(mirror(point, axis_start, axis_end))

    if symmetry == 'p6m':
        shape = 'triangle-right'
        atom_a_list, atom_b_list, unit_w, unit_h, radius_a, radius_b, angle_d, boundary_shape = unit_cell(image, radius_a, shape, symmetry, boundary=True, avoid_break_symmetry=True)                
        
        angle_d = 60 # fix the angle to 60 for translation unit cell
        repeat_w, repeat_h = 2*unit_w, 3*unit_h
        delta_w = round(repeat_h/math.tan(math.radians(angle_d)), 2)
      
        ## remove repeating atoms
        atom_a_unit_cell_list = remove_repeat_point(atom_a_list, radius_a)
        atom_b_unit_cell_list = remove_repeat_point(atom_b_list, radius_b)

        xa,ya = atom_a_unit_cell_list[0]
        copy_a, copy_b = atom_a_unit_cell_list.copy(), atom_b_unit_cell_list.copy()
        axis_start = xa, ya
        axis_end = xa, ya+unit_h
        for point in copy_a:
            atom_a_unit_cell_list.append(mirror(point, axis_start, axis_end))
        for point in copy_b:
            atom_b_unit_cell_list.append(mirror(point, axis_start, axis_end))  
        
        copy_a, copy_b = atom_a_unit_cell_list.copy(), atom_b_unit_cell_list.copy()
        center = xa, ya
        for point in copy_a:
            atom_a_unit_cell_list.append(rotate_xy(center, point, 120))
        for point in copy_b:
            atom_b_unit_cell_list.append(rotate_xy(center, point, 120))
        
        for point in copy_a:
            atom_a_unit_cell_list.append(rotate_xy(center, point, -120))
        for point in copy_b:
            atom_b_unit_cell_list.append(rotate_xy(center, point, -120))  
            
        copy_a, copy_b = atom_a_unit_cell_list.copy(), atom_b_unit_cell_list.copy()        
        center = (boundary_shape[0][0]+int(unit_w/2), boundary_shape[0][1]-int(unit_h/2))
        for point in copy_a:
            atom_a_unit_cell_list.append(rotate_xy(center, point, 180))
        for point in copy_b:
            atom_b_unit_cell_list.append(rotate_xy(center, point, 180))            
    

    if symmetry in ['p4m', 'p4g']:
        shape = 'triangle-right'
        atom_a_list, atom_b_list, unit_w, unit_h, radius_a, radius_b, angle_d, boundary_shape = unit_cell(image, radius_a, shape, symmetry, boundary=True, avoid_break_symmetry=True)        
        
#         print(boundary_shape)
        repeat_w, repeat_h = 2*unit_w, 2*unit_h
        delta_w = round(repeat_h/math.tan(math.radians(angle_d)), 2)

        ## remove repeating atoms
        atom_a_unit_cell_list = remove_repeat_point(atom_a_list, radius_a)
        atom_b_unit_cell_list = remove_repeat_point(atom_b_list, radius_b)
  
        copy_a, copy_b = atom_a_unit_cell_list.copy(), atom_b_unit_cell_list.copy()
        xa,ya = atom_a_unit_cell_list[0]
        
        axis_start = xa, ya  # top right corner of right triangle
        axis_end = xa-unit_w, ya+unit_h
        
        for point in copy_a:
            atom_a_unit_cell_list.append(mirror(point, axis_start, axis_end))
        for point in copy_b:
            atom_b_unit_cell_list.append(mirror(point, axis_start, axis_end))
        
        copy_a, copy_b = atom_a_unit_cell_list.copy(), atom_b_unit_cell_list.copy()
        if symmetry == 'p4m': center = xa, ya
        if symmetry == 'p4g': center = xa-unit_w, ya
        for point in copy_a:
            atom_a_unit_cell_list.append(rotate_xy(center, point, 90))
            atom_a_unit_cell_list.append(rotate_xy(center, point, 180))
            atom_a_unit_cell_list.append(rotate_xy(center, point, -90))
        for point in copy_b:
            atom_b_unit_cell_list.append(rotate_xy(center, point, 90))
            atom_b_unit_cell_list.append(rotate_xy(center, point, 180))
            atom_b_unit_cell_list.append(rotate_xy(center, point, -90))    
            

    # return generation parameters:
    
    if shape in ['square', 'rectangular', 'oblique', 'hexagonal']: 
        area = unit_w * unit_h
    if shape == 'rhombic' or shape[:8]=='triangle': 
        area = round(unit_w * unit_h / 2, 2)
        
    if symmetry in ['p1']: area_trans = 1*area
    if symmetry in ['p2', 'pm', 'pg', 'cm']: area_trans = 2*area
    if symmetry in ['pmm', 'pmg', 'pgg', 'cmm', 'p4']: area_trans = 4*area
    if symmetry in ['p3']: area_trans = 3*area
    if symmetry in ['p31m', 'p3m1', 'p6']: area_trans = 6*area
    if symmetry in ['p4m', 'p4g']: area_trans = 8*area
    if symmetry in ['p6m']: area_trans = 12*area
    

    atom_a_unit_cell_np = np.array(atom_a_unit_cell_list)
    atom_b_unit_cell_np = np.array(atom_b_unit_cell_list)

    
    
    # rectangular
    va = np.array((0, unit_w))
    vb = np.array((unit_h, int(round(1/np.tan(np.deg2rad(angle_d))*unit_h))))
    start_point = np.array([boundary_shape[0][1], boundary_shape[0][0]])
    
    if shape == 'rhombic':
        start_point = np.array([boundary_shape[3][1], boundary_shape[3][0]])
        va = np.array((int(round(-unit_h/2)), int(round(unit_w/2))))
        vb = np.array((int(round(unit_h/2)), int(round(unit_w/2))))
      
    elif shape[:8] == 'triangle':
        start_point = np.array([boundary_shape[2][1], boundary_shape[2][0]])
        vb = np.array((0, unit_w))
        
        if shape == 'triangle-right':
            va = np.array((-unit_h, unit_w))
        else:
            va = np.array((-unit_h, int(round(unit_w/2))))

    elif shape == 'oblique' or shape == 'hexagonal':
        va = np.array([boundary_shape[1][1], boundary_shape[1][0]]) - start_point
        vb = np.array([boundary_shape[3][1], boundary_shape[3][0]]) - start_point
    
    
    if symmetry == 'p1':
        t_va, t_vb = va, vb
            
    elif symmetry in ['p2', 'pm', 'pg']:
        t_va, t_vb = va, 2*vb
        if shape == 'triangle':
            t_va, t_vb = va, va*np.array([-1, 1])
            
    elif symmetry in ['cm', 'cmm']:
        t_va, t_vb = va, va*np.array([-1, 1])
        
    elif symmetry in ['pmm', 'pmg', 'p4']:
        t_va, t_vb = va*2, vb*2
        
    elif symmetry == 'pgg':
        start_point = start_point + va*np.array([1, 0])
        t_va, t_vb = vb, va*np.array([-2, 0])

    elif symmetry in ['p4m']:
        va = va*np.array([-1, 1])
        t_va, t_vb = va*np.array([2, 0]), vb*2
        start_point = start_point - t_va

    elif symmetry in ['p4g']: 
        t_va, t_vb = vb*2, va*np.array([-2, 0])
        start_point = start_point + 2*va -3*vb

    elif symmetry in ['p3']: 
        start_point = start_point + vb
        vb = -vb
        t_va, t_vb = vb*np.array((0, 2)), vb*np.array((3, -1))
        
    elif symmetry in ['p3m1']: 
        start_point = start_point - vb
        t_va, t_vb = va*np.array((1, 3)), va*np.array((-1, 3))

    elif symmetry in ['p31m', 'p6']: 
        t_va, t_vb = vb, va*np.array([3, 1])
        
    elif symmetry in ['p6m']: 
        t_va, t_vb = vb*2, va*np.array([3, 1])
    
    # center the boundary and atoms to center area
    start_point_0 = (image.shape[1]//2, image.shape[0]//3)
    diff = start_point_0 - start_point
    diff = np.array((diff[1], diff[0]))
    
    boundary_shape_0 = re_center_ts(boundary_shape[0], t_va, t_vb, image.shape)
    boundary_shape = np.array(boundary_shape) # x, y
    boundary_shape += diff
    atom_a_list += diff    
    atom_b_list += diff
    atom_a_unit_cell_list += diff
    atom_b_unit_cell_list += diff
    start_point += (diff[1], diff[0]) ########## ?reason for unit cell and whole image not match
    
    
    
# add rotation version

    # get all the points
    height_, width_ = height, width
    

    if rotation:
        r_angle = np.random.randint(0, 180)
        origin = int(width//2), int(height//2)
        
        # change order from h,w to x,y
        start_point = np.array((start_point[1], start_point[0]))
        va, vb = np.array((va[1], va[0])), np.array((vb[1], vb[0]))
        t_va, t_vb = np.array((t_va[1], t_va[0])), np.array((t_vb[1], t_vb[0]))
        
        # rotate vectors
        start_point = rotate_xy(origin, start_point, -r_angle)
        va, vb = rotate_xy((0,0), va, -r_angle), rotate_xy((0,0), vb, -r_angle)
        t_va, t_vb = rotate_xy((0,0), t_va, -r_angle), rotate_xy((0,0), t_vb, -r_angle)

        # change back to h,w
        start_point = np.array((start_point[1], start_point[0]))
        va, vb = np.array((va[1], va[0])), np.array((vb[1], vb[0]))
        t_va, t_vb = np.array((t_va[1], t_va[0])), np.array((t_vb[1], t_vb[0]))
        
        
        # rotate points
        for i, pa in enumerate(atom_a_unit_cell_list):
            atom_a_unit_cell_list[i] = rotate_xy(origin, pa, -r_angle)
        for i, pb in enumerate(atom_b_unit_cell_list):
            atom_b_unit_cell_list[i] = rotate_xy(origin, pb, -r_angle)
    
    
    # move ts to center
    middle_point = np.array((out_size[0]//2, out_size[1]//2))
    for j in range(100):
        dist = np.linalg.norm(start_point-middle_point)
        dist_new = dist+1

        n = 0
        while dist_new > dist:
            start_point_temp = start_point + random.choice([t_va, -t_va, t_vb, -t_vb])
            dist_new = np.linalg.norm(start_point_temp-middle_point)
            n += 1
            if n > 1000:
                start_point_temp = start_point
                break
        dist = dist_new
        dist_new = dist+1
        start_point = start_point_temp
    

    # draw atoms and boundary
    result_uc = draw_atoms_with_boundary(np.copy(image), radius_a, radius_b, atom_a_unit_cell_list, atom_b_unit_cell_list, boundary_shape)
    result_uc[result_uc < 0] = 0  # get rid of negative value   
    result_uc[result_uc > 1] = 1  # get rid of larger than 1 value   


    # get the whole image done
    atom_a_unit_cell_list = list(map(tuple, atom_a_unit_cell_list))
    atom_b_unit_cell_list = list(map(tuple, atom_b_unit_cell_list))
    
    
    atom_a_unit_cell_list_unique = get_unique_points_in_unit_cell(atom_a_unit_cell_list, t_va, t_vb, radius_a)
#     print('atom_a_unit_cell_list_unique', atom_a_unit_cell_list_unique)
    atom_b_unit_cell_list_unique = get_unique_points_in_unit_cell(atom_b_unit_cell_list, t_va, t_vb, radius_b)

    
    # new version  get all points function
    atom_a_all_list, atom_b_all_list = [], []

    for point in atom_a_unit_cell_list_unique:
        atom_a_all_list = atom_a_all_list + get_all_points_by_vector_new(point, out_size, t_va, t_vb, radius_a)
        
    for point in atom_b_unit_cell_list_unique:
        atom_b_all_list = atom_b_all_list + get_all_points_by_vector_new(point, out_size, t_va, t_vb, radius_b)
    
    atom_a_all, atom_b_all = np.array(atom_a_all_list), np.array(atom_b_all_list)
    boundary_shape = np.array(boundary_shape)


    # draw atoms and boundary
    result = draw_atoms_with_boundary(np.copy(image), radius_a, radius_b, atom_a_all, atom_b_all, boundary_shape)
    result[result < 0] = 0  # get rid of negative value   
    result[result > 1] = 1  # get rid of larger than 1 value   


    # convert str to int
    symmetry_dict = {'p1': 0, 'p2': 1, 'pm': 2, 'pg': 3, 'cm': 4, 'pmm': 5, 'pmg': 6, 'pgg': 7, 'cmm': 8, 
                     'p4': 9, 'p4m': 10, 'p4g': 11, 'p3': 12, 'p3m1': 13, 'p31m': 14, 'p6': 15, 'p6m': 16}
    symmetry_int = symmetry_dict[symmetry]

    shape_dict = { 'rectangular': 0, 'square': 0, 'oblique': 1, 'hexagonal': 1, 
                  'rhombic': 2, 'triangle': 3, 'triangle-120': 3, 'triangle-right': 3, 
                  'triangle-right-45': 3, 'triangle-right-60': 3}
    shape_int = shape_dict[shape]
    metadata = np.array((start_point, va, vb, t_va, t_vb, (symmetry_int, shape_int)))


    if show_step:
        verify_image_vector(result, start_point, t_va, t_vb)
        
    result = (result*255.).astype(np.uint8)
    result_uc = (result_uc*255.).astype(np.uint8)
    return result, result_uc, metadata
#     return result, result_uc, metadata, atom_a_unit_cell_list, atom_a_all, radius_a