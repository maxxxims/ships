import numpy as np


class CoordinateConversion:
    def __init__(self, x0, y0, z0, R_earth):
        self.x0 = x0
        self.y0 = y0
        self.z0 = z0
        self.R  = R_earth
        self.A  = self.calc_A()
        self.center_point = np.array([x0, y0, z0])


    def transform_local_image_to_GPDSK(self, x_coord: np.ndarray, y_coord: np.ndarray) -> tuple:
        assert len(x_coord) == len(y_coord)
        z_coord = np.zeros_like(x_coord)
        coords = np.array([x_coord, y_coord, z_coord], dtype=np.float64).T 
        new_coords = np.apply_along_axis(lambda x: self.A @ x, axis=1, arr=coords)
        x_new, y_new, z_new = new_coords.T
        return x_new, y_new, z_new


    def calc_A(self) -> np.ndarray:
        th = self.get_theta()
        phi = self.get_phi()
        A = np.array([
            [np.cos(th) / np.abs(np.cos(th)) * np.sin(phi),     np.sin(th) * np.cos(phi),       np.cos(th)*np.cos(phi)],
            [np.cos(th) / np.abs(np.cos(th)) * np.cos(phi),     np.sin(th) * np.sin(phi),       np.cos(th)*np.sin(phi)],
            [0,                                                 np.cos(th),                     np.sin(th)],
        ])
        return A

    def get_theta(self):
        return np.arcsin(self.z0 / self.R)
    
    def get_phi(self):
        if self.x0 == 0:
            return np.pi / 2
        return np.arctan(self.y0 / self.x0)
    


if __name__ == "__main__":
    reference_point = [0.0, 431785.69, 6385417.85]
    transformer = CoordinateConversion(reference_point[0], reference_point[1], reference_point[2], 6400000.0)
    x_coords = [800, 800]
    y_coords = [800, 800]
    coords = transformer.transform_local_image_to_GPDSK(x_coords, y_coords)
    print(coords)
