import numpy as np
def calc_area(p1,p2,p3):
    x_1s, y_1s = p1[0], p1[1]
    x_2s, y_2s = p2[0], p2[1]
    x_3s, y_3s = p3[0], p3[1]
    
    dists_1 = np.sqrt((x_2s - x_1s) ** 2 + (y_2s - y_1s) ** 2)
    dists_2 = np.sqrt((x_3s - x_2s) ** 2 + (y_3s - y_2s) ** 2)
    dists_3 = np.sqrt((x_1s - x_3s) ** 2 + (y_1s - y_3s) ** 2)
    s = (dists_1 + dists_2 + dists_3) /2.
    
    areas = np.sqrt(s * (s - dists_1) * (s - dists_2) * (s - dists_3))
    return areas

N = 1_000_000
R = 10
np.random.seed(10)
radiuses_1,angles_1 = R * np.random.random(N), 2 * np.pi * np.random.random(N)
radiuses_2,angles_2 = R * np.random.random(N), 2 * np.pi * np.random.random(N)
radiuses_3,angles_3 = R * np.random.random(N), 2 * np.pi * np.random.random(N)

x_1s, y_1s = radiuses_1 * np.sin(angles_1),radiuses_1 * np.cos(angles_1)
x_2s, y_2s = radiuses_2 * np.sin(angles_2),radiuses_2 * np.cos(angles_2)
x_3s, y_3s = radiuses_3 * np.sin(angles_3),radiuses_3 * np.cos(angles_3)

x_zeros,y_zeros = np.zeros(N), np.zeros(N)

true_area = calc_area(p1=[x_1s, y_1s], p2=[x_2s, y_2s], p3=[x_3s, y_3s])

area_calc_with_center = calc_area(p1=[x_zeros, y_zeros], p2=[x_2s, y_2s], p3=[x_3s, y_3s])+\
                        calc_area(p1=[x_1s, y_1s], p2=[x_zeros, y_zeros], p3=[x_3s, y_3s])+\
                        calc_area(p1=[x_1s, y_1s], p2=[x_2s, y_2s], p3=[x_zeros, y_zeros])
                        
res = np.mean(1.0*(np.abs(true_area - area_calc_with_center)<1e-9))

print(res)