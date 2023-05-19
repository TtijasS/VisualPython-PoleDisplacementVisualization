from vpython import vector, arrow, scene, box, color, rate, label, sphere, ring, sleep, cylinder
from math import radians, degrees, sin, cos, isclose, atan2
from keyboard import is_pressed
import quaternion as quat
import numpy as np
from numpy import quaternion

"""
Notes
I chose to represent vehicle in a 3D space with quaternions
Projection is calculated through previous --> current vectors (vector_angle, line 127)

Task 1. Projection of the post is displayed in 'up' quaternion
Task 2. Vehicle heading is displayed as 'Yaw' and as 'forward' quaternion
(both are printed)
data is read from data.txt file in this directory

Vehicle movement is visualized in vpython (displayed through browser)
-xyz stationary Cartesian axes are red, green and blue arrows
-Vehicle platform is represented as a box
-vehicle has it's own dynamic Cartesian system (orange, purple and cyan arrows)
-ground is represented as a thin box
"""


def visualize_data(file):
    prev_v = (0, 0)
    for line in file:
        try:
            t, x_mm, y_mm, roll, pitch = line.strip().split(",")
            x_mm, y_mm, roll, pitch = int(x_mm), int(y_mm), float(roll), float(pitch)
            my_vehicle.visualize_position(roll, pitch, prev_v, (x_mm, y_mm))
            sleep(2)
        except ValueError:
            print("Heading parameters/ Empty line")


def rotation_q(q, theta: float):
    """
    Calculates  q = [cos(theta/2)*w, sin(theta/2)*v]
    rot_quat is used in p' = q*p*q^-1 rotation formula
    :param q: unit quaternion representing axis of rotation
    :param theta: angle in degrees (to rotate around axis)
    :return: quaternion with applied rotation
    """
    theta_rad = radians(theta)
    w = cos(theta_rad / 2)
    s = sin(theta_rad / 2)
    return quaternion(w, s * q.x, s * q.y, s * q.z)


def normalize(v):
    # linalg.norm does a bunch of other steps that are redundant
    # mag = np.linalg.norm(quat.as_float_array(v))
    # calculation of vector magnitude
    mag = sum(i * i for i in quat.as_float_array(v))
    if mag == isclose(mag, 0, abs_tol = 1e-14):
        return v
    return v / mag


def rotate_quaternions(axis, theta, *quats):
    rot = rotation_q(axis, theta)
    rotated_quats = quat.rotate_vectors(rot, quat.as_vector_part(quats))
    return quat.from_vector_part(rotated_quats)


def vector_projection(a, b, s = 0., norm = True):
    """
    Vector projection of vector 'a' onto vector 'b'
    Math. notation: proj b(a)
    a1 = dot(a,b^)*b^
    (a1 - vector projection; b^ - unit vector)
    :param a: vector a
    :param b: vector b
    :param s: scalar component of vector 'u'. Used when 'u' is a unit vector, and you wish to define its length
    :param norm: if 'b' is already normed put norm=False
    :return: vector projection a1
    """
    if norm:
        # normalizes b
        b = b / np.linalg.norm(b)
    if s:
        a *= s
    return np.dot(a, b) * b


def scalar_projection(a, b, s = 0., norm = True):
    """
    Scalar projection of vector 'a' onto vector 'b'
    Math. notation: proj b(a)
    r = dot(a,b^)
    (r - scalar projection; b^ - unit vector)
    :param a: vector a
    :param b: vector b
    :param s: scalar component of vector 'u'. Used when 'u' is a unit vector and you wish to define its length
    :param norm: if 'b' is already normed put norm=False
    :return: vector projection a1
    """
    if norm:
        # normalizes b
        b = b / np.linalg.norm(b)
    if s:
        a *= s
    return np.dot(a, b)


def plane_projection(u, n, s = 0.):
    """
    Projection of vector 'u' onto the plane with a normal vector 'n'
    :param u: projected vector u
    :param n: vector normal to the plane... Example [0,1,0]
    :param s: scalar component of vector 'u'. Used when 'u' is a unit vector and you wish to define its length
    :return: vector projection on a plane 'P' with normal vector 'n'
    """
    if s:
        u *= s
    return u - vector_projection(u, n)


def move_vector(v1, v2):
    """
    Vector [v1(start) ----> v2(end) ] is moved to the coordinate system origin, keeping its magnitude and direction
    :param v1: vector 1 (previous vehicle position)
    :param v2: vector 2 (new vehicle position)
    :return: new vector moved to the origin, but with the same angle v1->v2
    """
    x1, y1 = v1
    x2, y2 = v2

    x, y = x2 - x1, y2 - y1
    return [x, y]


def vector_angle(v, second_v):
    """
        Takes two R^2 vectors v = [1,2] and outputs vector angle in degrees

        Explanation: starting coordinates v0 = [1,1], new coordinates v1 = [2,2],
        this f calculates the direction of the vector starting in v0 and ending in v1
        :param v: Starting coordinates of the vector
        :param second_v: End coordinates of the vector
        :return: Vector angle in degrees
        """
    x, y = move_vector(v, second_v)  # if there are 2 vectors
    return degrees(atan2(x, y))


class VehicleVisualization:
    def __init__(self):
        """
        A bunch of stuff from vpython so we can visualize our 'vehicle' in 3D space
        """
        # How far camera is positioned
        scene.range = 20
        # Background color
        scene.background = color.white
        # Defines scene forward vector
        scene.forward = vector(-1, -1, -1)
        # Scene window size
        scene.width = 800
        scene.height = 800

        # Axis quaternions
        self.front_q = quaternion(0, 1, 0, 0)
        self.up_q = quaternion(0, 0, 1, 0)
        self.side_q = quaternion(0, 0, 0, 1)
        self.always_up = quaternion(0, 0, 1, 0)

        # This is our static 3D Cartesian system, where we define xyz axes,
        # but the y is pointing up, since we are using quaternions for rotations
        self.x_arrow = arrow(shaftwidth = .1, color = color.red, opacity = .5, axis = vector(1, 0, 0), length = 15)
        # - x
        self._x_arrow = arrow(shaftwidth = .1, color = color.red, opacity = .5, axis = vector(-1, 0, 0), length = 15)

        self.y_arrow = arrow(shaftwidth = .1, color = color.blue, opacity = .5, axis = vector(0, 0, 1), length = 15)
        # - y
        self._y_arrow = arrow(shaftwidth = .1, color = color.blue, opacity = .5, axis = vector(0, 0, -1), length = 15)

        self.z_arrow = arrow(shaftwidth = .1, color = color.green, opacity = .5, axis = vector(0, 1, 0), length = 15)
        self._z_arrow = arrow(shaftwidth = .1, color = color.green, opacity = .5, axis = vector(0, -1, 0), length = 15)

        # This is our dynamic 3D Cartesian system... different colors
        self.front_arrow = arrow(shaftwidth = .1, color = color.orange, axis = vector(1, 0, 0), length = 2)
        self.side_arrow = arrow(shaftwidth = .1, color = color.purple, axis = vector(0, 1, 0))
        self.up_arrow = arrow(shaftwidth = .1, color = color.magenta, axis = vector(0, 0, 1))

        # Rotation rings
        self.roll_ring = ring(axis = vector(1, 0, 0), color = color.orange, thickness = .1)
        self.pitch_ring = ring(axis = vector(0, 1, 0), color = color.purple, thickness = .1)
        self.yaw_ring = ring(axis = vector(0, 0, 1), color = color.magenta, thickness = .1)

        # This represents the floor on which the vehicle is moving on
        self.floor = box(length = 60, width = 60, height = .05, opacity = .3, color = color.black)

        # Projection post and vector
        self.proj_vec = arrow(shaftwidth = .1, color = color.white, axis = vector(0, 0, 0))

        self.proj_sphere = sphere(radius = .1, pos = vector(0, 0, 0))

        # this represents a vehicle platform
        self.platform = box(length = 4, width = 2, height = .5, opacity = .4)
        # 1500mm post
        self.cylindr = cylinder(axis = self.up_arrow.axis, radius = .1, opacity = .3)

        # text labels
        self.x_label = label(pos = self.x_arrow.axis, text = "x")
        self._x_label = label(pos = self._x_arrow.axis, text = "-x")
        self.y_label = label(pos = self.y_arrow.axis, text = "y")
        self._y_label = label(pos = self._y_arrow.axis, text = "-y")
        self.proj_label = label(pos = self.proj_vec.axis, text = "", xoffset = 20)
        self.roll_label = label(pos = self.side_arrow.axis, text = "roll", xoffset = 20)
        self.pitch_label = label(pos = self.front_arrow.axis, text = "pitch", xoffset = 20)
        self.yaw_label = label(pos = self.up_arrow.axis, text = "yaw", xoffset = 20)
        self.gnss_label = label(pos = self.up_arrow.axis, text = "GNSS")
        self.info_label = label(pixel_pos = True, pos = vector(580, 60, 0), align = "left")

    def __str__(self):
        return f"forward: {self.front_q}\nside: {self.side_q}\nup: {self.up_q}"

    def test_keyboard(self):
        """
        You can move 'vehicle' using keyboard
        w s = pitch, a d = roll, q r = yaw, y x = yaw around [0,1,0] axis
        :return:
        """

        def correct_angle(theta):
            return -179 if theta == 181 \
                else 179 if theta == -181 \
                else theta

        roll, pitch, yaw = 0, 0, 0
        while True:
            rate(30)
            # Pitch
            if is_pressed("w"):
                self.front_q, self.up_q = rotate_quaternions(self.side_q, -1, self.front_q, self.up_q)
                pitch = correct_angle(pitch + 1)
            if is_pressed("s"):
                self.front_q, self.up_q = rotate_quaternions(self.side_q, 1, self.front_q, self.up_q)
                pitch = correct_angle(pitch - 1)

            # Roll
            if is_pressed("a"):
                self.up_q, self.side_q = rotate_quaternions(self.front_q, -1, self.up_q, self.side_q)
                roll = correct_angle(roll + 1)
            if is_pressed("d"):
                self.up_q, self.side_q = rotate_quaternions(self.front_q, 1, self.up_q, self.side_q)
                roll = correct_angle(roll - 1)

            # Yaw
            if is_pressed("q"):
                self.front_q, self.side_q = rotate_quaternions(self.up_q, 1, self.front_q, self.side_q)
                yaw = correct_angle(yaw + 1)
            if is_pressed("e"):
                self.front_q, self.side_q = rotate_quaternions(self.up_q, -1, self.front_q, self.side_q)
                yaw = correct_angle(yaw - 1)

            # Rotation around coordinate system up axis (in our case y)
            if is_pressed("y"):
                self.front_q, self.up_q, self.side_q = rotate_quaternions(self.always_up, 1, self.front_q, self.up_q,
                                                                          self.side_q)
                yaw = correct_angle(yaw + 1)
            if is_pressed("x"):
                self.front_q, self.up_q, self.side_q = rotate_quaternions(self.always_up, -1, self.front_q, self.up_q,
                                                                          self.side_q)
                yaw = correct_angle(yaw - 1)

            sn = 15  # scaling number
            post_size = 1500  # mm
            # SCALING STATIC VECTORS (x,y,z cartesian vectors)
            self.x_arrow.length = sn
            self._x_arrow.length = sn
            self.y_arrow.length = sn
            self._y_arrow.length = sn
            self.z_arrow.length = sn
            self._z_arrow.length = sn

            # DYNAMIC VECTORS, VEHICLE, RINGS, PROJECTION

            # Arrows front, side, up
            arrow_len = 5
            ring_radius = arrow_len
            front_axis = vector(self.front_q.x, self.front_q.y, self.front_q.z)
            side_axis = vector(self.side_q.x, self.side_q.y, self.side_q.z)
            up_axis = vector(self.up_q.x, self.up_q.y, self.up_q.z)

            # Arrows
            self.front_arrow.axis = front_axis
            self.front_arrow.length = arrow_len
            self.up_arrow.axis = up_axis
            self.up_arrow.length = arrow_len
            self.side_arrow.axis = side_axis
            self.side_arrow.length = arrow_len

            # Cylinder (vehicle post)
            self.cylindr.axis = up_axis * sn

            # Vehicle platform
            self.platform.axis = front_axis
            self.platform.up = up_axis
            self.platform.width = sn // 2
            self.platform.length = sn

            # Projection vector
            up_vec = quat.as_vector_part(self.up_q)
            # Because vpython defines y as up axis, we will treat y as z
            px, py, pz = plane_projection(up_vec, np.array([0, 1, 0]), s = post_size)
            # Again we scale down
            scaling_n = 10 ** (len(str(post_size)) - 2)
            self.proj_vec.axis = vector(px / scaling_n, py / scaling_n, pz / scaling_n)

            # Rings
            self.roll_ring.axis = front_axis
            self.roll_ring.radius = ring_radius
            self.pitch_ring.axis = side_axis
            self.pitch_ring.radius = ring_radius
            self.yaw_ring.axis = up_axis
            self.yaw_ring.radius = ring_radius

            # Labels
            self.roll_label.pos = self.side_arrow.axis
            self.roll_label.text = f"roll: {roll}°"
            self.pitch_label.pos = self.front_arrow.axis
            self.pitch_label.text = f"pitch: {pitch}°"
            self.yaw_label.pos = self.up_arrow.axis
            self.yaw_label.text = f"yaw: {yaw}°"
            self.proj_label.pos = self.proj_vec.axis
            self.proj_label.text = f"x: {round(px)}mm, y: {round(pz)}mm"
            self.gnss_label.pos = self.cylindr.axis

    def visualize_position(self, roll, pitch, *v, yaw = .0, post_size = 1500):
        """
        You can leave out yaw and input two vectors instead
        :param roll: in deg
        :param pitch: in deg
        :param v: v1, v2 where v1 = previous x,y coordinates, v2 = current coordinates
        :param yaw: in deg
        :param post_size: length of GNSS module post (default 1500)
        :return: vehicle visualized in browser with vpython
        """

        # SCALING DOWN POST SIZE
        def scale_down(ps):
            """
            Post size divided by its number of digits minus 2.
            example ps = 1500, ps is a 4 digit number so
            returns 1500 // 10**2
            :param ps: post size
            :return: scaled post size
            """
            i = len(str(ps))
            if i > 2:
                n = 10 ** (i - 2)
                i = int(ps) // n
            return i

        sn = scale_down(post_size)  # Scaling number
        if v:
            yaw = round(vector_angle(v[0], v[1]), 2)

            # PRESET
        # Default starting positions of vehicle quats
        self.front_q = quaternion(0, 1, 0, 0)
        self.up_q = quaternion(0, 0, 1, 0)
        self.side_q = quaternion(0, 0, 0, 1)

        # ROTATING THE 3 MAIN DYNAMIC VECTORS
        # yaw
        self.front_q, self.side_q = rotate_quaternions(self.up_q, -yaw, self.front_q, self.side_q)
        # pitch
        self.front_q, self.up_q = rotate_quaternions(self.side_q, pitch, self.front_q, self.up_q)
        # roll
        self.up_q, self.side_q = rotate_quaternions(self.front_q, roll, self.up_q, self.side_q)

        # SCALING STATIC VECTORS (x,y,z cartesian vectors)
        self.x_arrow.length = sn
        self._x_arrow.length = sn
        self.y_arrow.length = sn
        self._y_arrow.length = sn
        self.z_arrow.length = sn
        self._z_arrow.length = sn

        # DYNAMIC VECTORS, VEHICLE, RINGS, PROJECTION

        # Arrows front, side, up
        arrow_len = sn // 3
        ring_radius = arrow_len
        front_axis = vector(self.front_q.x, self.front_q.y, self.front_q.z)
        side_axis = vector(self.side_q.x, self.side_q.y, self.side_q.z)
        up_axis = vector(self.up_q.x, self.up_q.y, self.up_q.z)

        # Arrows
        self.front_arrow.axis = front_axis
        self.front_arrow.length = arrow_len
        self.up_arrow.axis = up_axis
        self.up_arrow.length = arrow_len
        self.side_arrow.axis = side_axis
        self.side_arrow.length = arrow_len

        # Cylinder (vehicle post)
        self.cylindr.axis = up_axis * sn

        # Vehicle platform
        self.platform.axis = front_axis
        self.platform.up = up_axis
        self.platform.width = sn // 2
        self.platform.length = sn

        # Projection vector
        up_vec = quat.as_vector_part(self.up_q)
        # Because vpython defines y as up axis, we will treat y as z
        px, py, pz = plane_projection(up_vec, np.array([0, 1, 0]), s = post_size)
        # Again we scale down
        scaling_n = 10 ** (len(str(post_size)) - 2)
        self.proj_vec.axis = vector(px / scaling_n, py / scaling_n, pz / scaling_n)

        # Rings
        self.roll_ring.axis = front_axis
        self.roll_ring.radius = ring_radius
        self.pitch_ring.axis = side_axis
        self.pitch_ring.radius = ring_radius
        self.yaw_ring.axis = up_axis
        self.yaw_ring.radius = ring_radius

        # Labels
        self.roll_label.pos = self.side_arrow.axis
        self.roll_label.text = f"roll: {roll}°"
        self.pitch_label.pos = self.front_arrow.axis
        self.pitch_label.text = f"pitch: {pitch}°"
        self.yaw_label.pos = self.up_arrow.axis
        self.yaw_label.text = f"yaw: {yaw}°"
        self.proj_label.pos = self.proj_vec.axis
        px_round, py_round = round(px), round(pz)  # not a mistake, [x,y,z] in vpython is [front, up, side]
        self.proj_label.text = f"proj [{px_round}, {py_round}]"
        self.gnss_label.pos = self.cylindr.axis

        try:
            # GNSS pos = x_mm, y_mm
            # GNSS proj = projection vector on a plane
            # Vehicle pos = GNSS pos - GNSS proj
            corrected_x, corrected_y = np.array(v[1]) - np.array([px_round, py_round])
            self.info_label.text = f"GNSS pos: [{v[1][0]}, {v[1][1]}]\n" \
                                   f"GNSS proj:  [{px_round}, {py_round}]\n" \
                                   f"Vehicle pos: [{corrected_x}, {corrected_y}]"
            print(self.info_label.text)
            print(f"Yaw: {yaw}°\n")

        except IndexError:
            self.info_label.text = f"GNSS proj:  [{px_round}, {py_round}]"
            print(self.info_label.text)
            print(f"Yaw: {yaw}°\n")


# Creates a new vehicle object, that can be visualized
my_vehicle = VehicleVisualization()

# Enables testing visualization with keyboard keys = [w, s, a, d, q, r, y, x]
"""For keyboard testing uncomment this"""
my_vehicle.test_keyboard()

"""and comment bellow"""
# data_file = open(r"./data.txt", "r")
# visualize_data(data_file)

# Testing on specific angles
# roll, pitch, yaw = 30, -30, 135
# my_vehicle.visualize_position(roll, pitch, yaw)
