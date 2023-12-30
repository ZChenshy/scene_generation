from trajectory import get_extrinsics,_rot_xyz

def get_extrinsics_matrix(R,T):
    """Given the rotation angle (represented by the radius) and the camera's position, return an extrinsics matrix.
    Args:
        rx: positive:up negatative:down
        ry: positive:left negatative:right
        rz: It is not recommended to modify this parameter. 
    """
    rx,ry,rz = R
    extrinsics_matrix = get_extrinsics(_rot_xyz(rx,ry,rz),T)
    return extrinsics_matrix

def get_intrinsic(fovx,fovy):
    pass
