# Python3 program to draw solid-colored
# image using numpy.zeroes() function
import numpy as np
import cv2
import pandas as pd
import torch
# import pyyaml module
import yaml
from yaml.loader import SafeLoader
import os
import time


START_TIME = time.time()

SEED = 42     
INT = np.int64
FLOAT = np.float64
UINT  = np.uint8
TOTAL_NUMBER_OF_SHAPES = 10
BACKGROUND_COLOR = 0 # black = 0 gray = 128 white = 255
SHAPE_THICKNESS = 2 #Thickness of -1 px will fill the rectangle shape by the specified color.



# 2 bits
COLOR_LIST = ['blue', 'green', 'red', 'white']
# 2 bits
Y_CENTER_SPACE = [0., 0.25, 0.5, 0.75] # 0, 1/4, 2/4, 3/4
# 2 bits
X_CENTER_SPACE = [0., 0.25, 0.5, 0.75] # 0, 1/4, 2/4, 3/4
# 2 bits
b_CENTER_SPACE = [0.0625, 0.125 , 0.1875, 0.25] # 1/16, 2/16, 3/16, 4/16
# 2 bits
a_CENTER_SPACE = [0.0625, 0.125 , 0.1875, 0.25] # 1/16, 2/16, 3/16, 4/16
# 2 bits
alpha_CENTER_SPACE = [30, 45, 60, 90]
# 0 bits
FILL_NOFILL = [-1, SHAPE_THICKNESS]

# calculate the number of bits required for exactly one shape
number_of_bits_required_for_one_shape = sum([np.log2(1.0 * len(space))\
                                            for space in \
                                            [COLOR_LIST,\
                                            Y_CENTER_SPACE,\
                                            X_CENTER_SPACE,\
                                            b_CENTER_SPACE,\
                                            a_CENTER_SPACE,\
                                            alpha_CENTER_SPACE,\
                                            FILL_NOFILL]])

# calculate the number of bits required for all shapes combined
number_of_bits_required_for_one_image = TOTAL_NUMBER_OF_SHAPES * number_of_bits_required_for_one_shape

# which shape features will be changed
# i.e. columns in pandas DF that will show the histograms of randomly distributed features
COLUMNS = [ #'shape_id',\
            'shape_name',\
            'a',\
            'b',\
            'alpha',\
            'shape_center_x',\
            'shape_center_y',\
            'color',\
            #'shape_thickness'\
            ]

class ImagesGenerator:
    def __init__(self) -> None:
        pass
    def __str__(self) -> str:
        pass

class SuperClassShape:
    def __init__(self, kwargs) -> None:
        # image info (to be implemented)
        # self.image_handle = kwargs['image_handle']
        
        # shape info
        # self.shape_handle = kwargs['shape_handle']
        self.shape_id = kwargs['shape_id']
        
        # random parameters to be changed/generated
        self.shape_center_x, self.shape_center_y = kwargs['shape_center_x'], kwargs['shape_center_y']
        self.shape_rotation_angle = kwargs['shape_rotation_angle']
        self.shape_scale_size = kwargs['shape_scale_size']
        self.shape_color = kwargs['shape_color']# red color in BGR = (0, 0, 255)
        self.shape_thickness = kwargs['shape_thickness']# 5 = # Line thickness of 5 px
        
        
    def __str__(self) -> str:
        # print out id of an shape and its name
        return f"Shape id = {self.shape_id}\n" + f"Shape name = {self.shape_name}"
    
    def draw_(self, path, image = None) -> None:
        raise NotImplementedError("This is abstract class.")
    
class Ellipse(SuperClassShape): # elipsa
    def __init__(self, kwargs) -> None:
        super().__init__(kwargs)
        self.shape_name = kwargs['shape_name'] # 'Ellipse'
        self.a, self.b, self.alpha = kwargs['a'], kwargs['b'], kwargs['alpha']
    def draw_(self, path, image = None) -> None:
        # Reading an image in default mode
        if image == None:
            image = cv2.imread(path)
        center_coordinates = (self.shape_center_x, self.shape_center_y)
        axesLength = (self.a, self.b)
        # (0-360) to draw a full Ellipse;
        # (0-180) to draw a half Ellipse;
        angle, startAngle,endAngle = self.alpha, 0 ,360
        color = self.shape_color#(0, 0, 255) # Red color in BGR
        image = cv2.ellipse(image, center_coordinates, axesLength,
                angle, startAngle, endAngle, color, self.shape_thickness)
        
        cv2.imwrite(path, image)
        #image = cv2.ellipse(image, center, axes, angle, 0., i, (0,255,0))
        
class Polygon(SuperClassShape): # mnogougao
    def __init__(self, kwargs) -> None:
        super().__init__(kwargs)
        self.shape_name = kwargs['shape_name'] # 'Polygon'
        self.l, self.n = kwargs['l'], kwargs['n']
    
    
class Parallelogram(SuperClassShape): # paralelogram - rectangle
    def __init__(self, kwargs) -> None:
        super().__init__(kwargs)
        self.shape_name = kwargs['shape_name'] #'Parallelogram'
        self.a, self.b, self.alpha = kwargs['a'], kwargs['b'], kwargs['alpha']
        
    def draw_(self, path, image = None) -> None:
        # Reading an image in default mode
        if image == None:
            image = cv2.imread(path)
        color = self.shape_color#(0, 0, 255) # Red color in BGR
        thickness = self.shape_thickness
        delta_x = None
        if self.alpha == 90:
            # Rectangle
            delta_x = 0
        else:
            # Parallelogram
            delta_x = int(round(self.b / (2.0*np.tanh(self.alpha))))
        
        upper_left_coords = (self.shape_center_x - self.a//2 + delta_x, \
                            self.shape_center_y - self.b//2)
        down_right_coords = (self.shape_center_x + self.a//2 + delta_x, \
                            self.shape_center_y + self.b//2)
        if self.alpha == 90:
            # Rectangle
            image = cv2.rectangle(image, upper_left_coords, down_right_coords, color, thickness)
        elif 0 < self.alpha and self.alpha < 90:
            # Parallelogram
            # nodes of Parallelogram =
            # A(lower left), B(lower right), C(upper right), D(upper left)
            D = [upper_left_coords[0], upper_left_coords[1]]
            C = [self.a + D[0] , D[1]]
            A = [D[0] - 2*delta_x , D[1] + self.b]             
            B = [self.a + A[0], A[1]]
            
            # Polygon corner points coordinates
            pts = np.array([D, A, B,C],np.int32).reshape((-1, 1, 2))     
            # example of pts
            # array([[[ 38, -50]],
            #       [[-37,  25]],
            #       [[ 63,  25]],
            #       [[138, -50]]], dtype=int32)       
            
            isClosed = True
            if thickness >= 0:
                image = cv2.polylines(image, [pts], isClosed, color, thickness)
            else:
                image = cv2.polylines(image, [pts], isClosed, color, 2)
                cv2.fillPoly(image, [pts], color)
            
            
        else:
            assert(False, "Alpha has to be between 0 and 90!" )
        cv2.imwrite(path, image)
class ShapeList:
    def __init__(self) -> None:
        self.TOTAL_NUMBER_OF_SHAPES = TOTAL_NUMBER_OF_SHAPES
        self.shape_list : SuperClassShape = [None] * self.TOTAL_NUMBER_OF_SHAPES
        
    
    def create_and_add_shape(self, kwargs_shape : dict) -> SuperClassShape:
        shape = None
        if kwargs_shape['shape_name'] == 'Ellipse':
            shape = Ellipse(kwargs_shape)
        elif kwargs_shape['shape_name'] == 'Polygon':
            shape = Polygon(kwargs_shape)
        elif kwargs_shape['shape_name'] == 'Parallelogram':
            shape = Parallelogram(kwargs_shape)
        else:
            assert(False, f"There is no shape of type = {kwargs_shape['shape_name']}.")
        self.shape_list.append(shape)    
        return shape
    
    
class GeneratedImage:
    def __init__(self, kwargs) -> None:
        
        # parsing info regarding the location of the file
        self.file_path : str      = kwargs["file_path"]
        self.file_name  : str     = kwargs["file_name"]
        self.file_version : str   = kwargs["file_version"] if "file_version" in kwargs else "X"
        self.file_extension : str = kwargs["file_extension"]
        
        # parsing info regarding the height, width and channel number
        for f in ['H', 'W', 'C']:
            assert(f in kwargs and kwargs[f] > 0 and type(kwargs[f]) == INT, f + ' must be defined and nonnegative integer for GeneratedImage!')
        self.H, self.W, self.C = kwargs['H'], kwargs['W'], kwargs['C']
        
        # parsing info regarding the content of the image
        # 'background_color'
        self.image_objects : dict   = kwargs['image_objects'] if "image_objects" in kwargs else {}

        self.TOTAL_NUMBER_OF_SHAPES = TOTAL_NUMBER_OF_SHAPES
        self.all_shapes_on_image = [None] * self.TOTAL_NUMBER_OF_SHAPES
        
    def generate_image(self, new_file_version = None) -> None:
        assert(self.image_objects['background_color'].dtype == UINT)
        # Creating a black image with 3 channels; RGB and unsigned int datatype
        self.image_objects['background_image'] = self.image_objects['background_color'] * np.ones((self.H, self.W, self.C), dtype = "uint8")
        cv2.imwrite(self.get_full_path(new_file_version), self.image_objects['background_image'])

    def get_full_path(self, new_file_version = None) -> str:
        return  self.file_path + \
                self.file_name + \
                (self.file_version if new_file_version == None else new_file_version) \
                + self.file_extension
    
    def add_shape(self, shape : SuperClassShape) -> None:
        shape.draw_(self.get_full_path())
    
    def add_shape_from_list(self, index_ : int, list_of_shapes : ShapeList) -> None:
        shape : SuperClassShape = list_of_shapes.shape_list[index_]
        shape.draw_(self.get_full_path())
        
    def draw_grid_on_image(self, X_coors, Y_coors, draw_lines = True, draw_circles = True) -> None:
        #image path
        image_path = self.get_full_path()

        # White color in BGR
        color = (255, 255 , 255)

        # Line thickness of 1 px
        thickness = 1

        if draw_lines:
            # drawing a grid (line by line)
            for x_coor in X_CENTER_SPACE_np:
                # start and end of the line
                start_point = (x_coor, np.min(Y_CENTER_SPACE_np))
                end_point = (x_coor, np.max(Y_CENTER_SPACE_np))
                image = cv2.line( cv2.imread(image_path), start_point, end_point, color, thickness)
                cv2.imwrite(image_path, image)
            
            # drawing a grid (line by line)
            for y_coor in Y_CENTER_SPACE_np:
                # start and end of the line
                start_point = (np.min(X_CENTER_SPACE_np), y_coor)
                end_point = (np.max(X_CENTER_SPACE_np), y_coor)
                image = cv2.line( cv2.imread(image_path), start_point, end_point, color, thickness)
                cv2.imwrite(image_path, image)
        
        if draw_circles:    
            for x_coor in X_CENTER_SPACE_np:
                for y_coor in Y_CENTER_SPACE_np:
                    # Center coordinates
                    center_coordinates = (x_coor, y_coor)
                    # Radius of circle
                    radius = 10
                    # CYAN color in BGR 
                    color = (255, 255, 0)# (255, 255, 255)
                    # Line thickness of -1 px
                    thickness = -1
                    # Using cv2.circle() method
                    # Draw a circle of red color of thickness -1 px
                    image = cv2.circle(cv2.imread(image_path), center_coordinates, radius, color, thickness)
                    cv2.imwrite(image_path, image)
        
        

np.random.seed(SEED)

current_working_absoulte_path = '/home/novakovm/iris/MILOS'
os.chdir(current_working_absoulte_path)

milos_config_path = 'milos_config.yaml'
# Open the file and load the file
with open(milos_config_path) as f:
    data = yaml.load(f, Loader=SafeLoader)

file_info_dict : dict = {key:val for one_info in data['file_info'] for key,val in one_info.items()}

# obj info

file_info_dict['image_objects'] = {'background_color' : np.array([BACKGROUND_COLOR], dtype=UINT)}

first_generated_image = GeneratedImage(file_info_dict)

first_generated_image.generate_image()

shape_info_dict : dict = {key:val for one_info in data['shape_info'] for key,val in one_info.items()}

shape_info_dict['shape_ids'] = np.arange(1,TOTAL_NUMBER_OF_SHAPES)

list_of_shapes = ShapeList()

# coordinates of all possible centers of shapes
Y_CENTER_SPACE_np = np.round(np.array(Y_CENTER_SPACE) * file_info_dict['H'], 0).astype(int)
X_CENTER_SPACE_np = np.round(np.array(X_CENTER_SPACE) * file_info_dict['W'], 0).astype(int)

# lengths of all possible dimensions of shapes
b_CENTER_SPACE_np = np.round(np.array(b_CENTER_SPACE) * file_info_dict['H'], 0).astype(int)
a_CENTER_SPACE_np = np.round(np.array(a_CENTER_SPACE) * file_info_dict['W'], 0).astype(int)

# angle of skewness (for Parallelogram) and rotation (for Ellipse)
alpha_CENTER_SPACE_np = np.array(alpha_CENTER_SPACE).astype(int)

# fills or no fills
FILL_NOFILL_np = np.array(FILL_NOFILL).astype(int)

first_generated_image.draw_grid_on_image(X_coors=X_CENTER_SPACE_np, Y_coors=Y_CENTER_SPACE_np)       

all_shapes_variable_data = {}
for c in COLUMNS:
    all_shapes_variable_data[c] = []


                

for i in range(TOTAL_NUMBER_OF_SHAPES):
    kwargs_shape = {}
    
    # image info (to be implemented)
    #kwargs_shape['image_handle'] = None
    
    # shape info
    #kwargs_shape['shape_handle'] = None
    
    kwargs_shape['shape_id'] = i
    
    
    # random parameters to be changed/generated
    kwargs_shape['shape_center_x'] = np.random.choice(X_CENTER_SPACE_np, size = 1)[0]
    kwargs_shape['shape_center_y'] = np.random.choice(Y_CENTER_SPACE_np, size = 1)[0]
    
    kwargs_shape['shape_rotation_angle'] = None
    kwargs_shape['shape_scale_size'] = None
    
    color = np.random.choice( COLOR_LIST, size = 1)
    if color == 'blue':
        kwargs_shape['shape_color'] = (255, 0, 0) # BGR code
    elif color == 'green':
        kwargs_shape['shape_color'] = (0, 255, 0) # BGR code
    elif color == 'red':
        kwargs_shape['shape_color'] = (0, 0, 255) # BGR code
    elif color == 'white':
        kwargs_shape['shape_color'] = (255, 255 , 255) # BGR code

    
    kwargs_shape['shape_thickness'] = np.random.choice(FILL_NOFILL_np, size = 1)[0] # 5 = # Line thickness of 5 px
    
    # shape specific fields
    kwargs_shape['shape_name'] = np.random.choice(['Ellipse','Parallelogram'], size=1)[0] # print(np.random.choice(prog_langs, size=10, replace=True, p=[0.3, 0.5, 0.0, 0.2]))
    
    
    
    if kwargs_shape['shape_name'] == "Ellipse":
        kwargs_shape['a'] = np.random.choice(a_CENTER_SPACE_np, size = 1)[0]
        kwargs_shape['b'] = np.random.choice(b_CENTER_SPACE_np, size = 1)[0]
        kwargs_shape['alpha'] = np.random.choice(alpha_CENTER_SPACE_np, size = 1)[0] #0
        
    elif kwargs_shape['shape_name'] == "Parallelogram":
        kwargs_shape['a'] = np.random.choice(a_CENTER_SPACE_np, size = 1)[0]
        kwargs_shape['b'] = np.random.choice(b_CENTER_SPACE_np, size = 1)[0]
        kwargs_shape['alpha'] = np.random.choice(alpha_CENTER_SPACE_np, size = 1)[0] #90
        
    else:
        assert(False, 'The shape must be Ellipse or Parallelogram!')
    
    # fill in pandas df
    for c in COLUMNS:
        all_shapes_variable_data[c].append(color[0] if c == 'color' else kwargs_shape[c])
    
    new_shape = list_of_shapes.create_and_add_shape(kwargs_shape)
    first_generated_image.add_shape(shape=new_shape)
    #first_generated_image.add_shape_from_list(index_=i, list_of_shapes=list_of_shapes)
    
# create pandas df
DF_all_shapes_variable_data = pd.DataFrame.from_dict(all_shapes_variable_data)
figure_all_shapes_variable_data = DF_all_shapes_variable_data.hist()[0][0].get_figure()
figure_all_shapes_variable_data.tight_layout()
figure_all_shapes_variable_data.savefig('DATA/shapes_stats.png')
 

# Allows us to see image
# until closed forcefully
cv2.waitKey(0)
cv2.destroyAllWindows()

print(f"Number of bits for one SHAPE = {number_of_bits_required_for_one_shape}b.")
print(f"Number of SHAPEs = {TOTAL_NUMBER_OF_SHAPES}.")
print(f"Number of bits for one IMAGE = {number_of_bits_required_for_one_image}b.")
print(f"Total number of seconds that the program runs = {round(time.time() - START_TIME)} sec.")
