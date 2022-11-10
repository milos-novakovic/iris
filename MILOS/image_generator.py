# Python3 program to draw solid-colored
# image using numpy.zeroes() function
import numpy as np
import cv2
# import pyyaml module
import yaml
from yaml.loader import SafeLoader
import os

INT = np.int64
FLOAT = np.float64
UINT  = np.uint8
TOTAL_NUMBER_OF_SHAPES = 20
BACKGROUND_COLOR = 0 # black = 0 gray = 128 white = 255
SHAPE_THICKNESS = 2 #Thickness of -1 px will fill the rectangle shape by the specified color.


COLOR_LIST = ['blue', 'green', 'red', 'white']

Y_CENTER_SPACE = [0., 0.25, 0.5, 0.75] # 0, 1/4, 2/4, 3/4
X_CENTER_SPACE = [0., 0.25, 0.5, 0.75] # 0, 1/4, 2/4, 3/4

b_CENTER_SPACE = [0.    , 0.0625, 0.125 , 0.1875] # 0, 1/16, 2/16, 3/16
a_CENTER_SPACE = [0.    , 0.0625, 0.125 , 0.1875] # 0, 1/16, 2/16, 3/16
alpha_CENTER_SPACE = [30, 45, 60, 90]

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
        self.a, self.b = kwargs['a'], kwargs['b']
    def draw_(self, path, image = None) -> None:
        # Reading an image in default mode
        if image == None:
            image = cv2.imread(path)
        center_coordinates = (self.shape_center_x, self.shape_center_y)
        axesLength = (self.a, self.b)
        angle = 0#self.shape_rotation_angle
        startAngle = 0
        endAngle = 360
        color = self.shape_color#(0, 0, 255) # Red color in BGR
        thickness = self.shape_thickness
        # Using cv2.ellipse() method
        # Draw a ellipse with red line borders of thickness of 5 px
        image = cv2.ellipse(image, center_coordinates, axesLength,
                angle, startAngle, endAngle, color, thickness)
        
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
        
        #rectangle
        self.alpha = 90
        
    def draw_(self, path, image = None) -> None:
        # Reading an image in default mode
        if image == None:
            image = cv2.imread(path)
        upper_left_coords = (self.shape_center_x - self.b//2, self.shape_center_y - self.a//2)
        down_right_coords = (self.shape_center_x + self.b//2, self.shape_center_y + self.a//2)
        color = self.shape_color#(0, 0, 255) # Red color in BGR
        thickness = self.shape_thickness
        
        if self.alpha == 90:
            # rectangle
            image = cv2.rectangle(image, upper_left_coords, down_right_coords, color, thickness)
        
        elif 0 < self.alpha and self.alpha < 90:
            # nodes of Parallelogram
            A = [upper_left_coords[0] , upper_left_coords[1] + self.b]             
            B = [self.a + A[0], A[1]]
            D = [upper_left_coords[0] + self.b * 1.0 / np.tanh(self.alpha), upper_left_coords[1]]
            C = [self.a + D[0] , D[1]]

            # Polygon corner points coordinates
            pts = np.array([D, A, B,C],np.int32).reshape((-1, 1, 2))            
            isClosed = True
            
            # Using cv2.polylines() method 
            image = cv2.polylines(image, [pts], isClosed, color, thickness)
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
        # 
        self.image_objects : dict   = kwargs['image_objects'] if "image_objects" in kwargs else {}

        self.TOTAL_NUMBER_OF_SHAPES = TOTAL_NUMBER_OF_SHAPES
        self.all_shapes_on_image = [None] * self.TOTAL_NUMBER_OF_SHAPES
        
    def generate_image(self, new_file_version = None) -> None:
        assert(self.image_objects['background_color'].dtype == UINT)
        
        # Creating a black image with 3 channels
        # RGB and unsigned int datatype
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
        
SEED = 42     
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


Y_CENTER_SPACE_np = np.round(np.array(Y_CENTER_SPACE) * file_info_dict['H'], 0).astype(int)
X_CENTER_SPACE_np = np.round(np.array(X_CENTER_SPACE) * file_info_dict['W'], 0).astype(int)


#Y_CENTER_SPACE = np.arange(0,file_info_dict['H'], file_info_dict['H'] // 4 )
#X_CENTER_SPACE = np.arange(0,file_info_dict['W'], file_info_dict['W'] // 4 )

b_CENTER_SPACE_np = np.round(np.array(b_CENTER_SPACE) * file_info_dict['H'], 0).astype(int) #np.arange(0,file_info_dict['H'], file_info_dict['H'] // 4 )
a_CENTER_SPACE_np = np.round(np.array(a_CENTER_SPACE) * file_info_dict['W'], 0).astype(int) #np.arange(0,file_info_dict['W'], file_info_dict['W'] // 4 )
alpha_CENTER_SPACE_np = np.array(alpha_CENTER_SPACE).astype(int)

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

    kwargs_shape['shape_thickness'] = SHAPE_THICKNESS# 5 = # Line thickness of 5 px
    
    # shape specific fields
    kwargs_shape['shape_name'] = np.random.choice(['Ellipse','Parallelogram'], size=1)[0] # print(np.random.choice(prog_langs, size=10, replace=True, p=[0.3, 0.5, 0.0, 0.2]))
    
    if kwargs_shape['shape_name'] == "Ellipse":
        kwargs_shape['a'] = np.random.choice(a_CENTER_SPACE_np, size = 1)[0]
        kwargs_shape['b'] = np.random.choice(b_CENTER_SPACE_np, size = 1)[0]
        
    elif kwargs_shape['shape_name'] == "Parallelogram":
        kwargs_shape['a'] = np.random.choice(a_CENTER_SPACE_np, size = 1)[0]
        kwargs_shape['b'] = np.random.choice(b_CENTER_SPACE_np, size = 1)[0]
        kwargs_shape['alpha'] = np.random.choice(alpha_CENTER_SPACE_np, size = 1)[0] #90
    
    
    new_shape = list_of_shapes.create_and_add_shape(kwargs_shape)
    first_generated_image.add_shape(shape=new_shape)
    #first_generated_image.add_shape_from_list(index_=i, list_of_shapes=list_of_shapes)
    
    
     
 
'''

file_full_str = file_path + file_name + file_extension

H,W,C= 600,800,3
# Creating a black image with 3 channels
# RGB and unsigned int datatype
img = 128*np.ones((H, W, C), dtype = "uint8")
cv2.imwrite(file_full_str, img)


# Creating a black image with 3
# channels RGB and unsigned int datatype
img = np.zeros((400, 400, 3), dtype = "uint8")
# Creating rectangle
cv2.rectangle(img, (30, 30), (300, 200), (0, 255, 0), 5)
cv2.imwrite(file_full_str, img)

# Creating a black image with 3
# channels RGB and unsigned int datatype
img = np.zeros((400, 400, 3), dtype = "uint8")
  
# Creating circle
cv2.circle(img, (200, 200), 80, (255, 0, 0), 3)
  
cv2.imshow('dark', img)

'''
# Allows us to see image
# until closed forcefully
cv2.waitKey(0)
cv2.destroyAllWindows()