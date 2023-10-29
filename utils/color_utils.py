import PIL
import math
from PIL import Image
import colormath
from colormath.color_objects import LabColor, sRGBColor
from colormath.color_conversions import convert_color
from colormath import color_diff_matrix
import numpy as np

#def create_color_image(hex_code, image_path, width=100, height=100):
#    rgb = hex_to_rgb(hex_code)
#    image = Image.new('RGB', (width, height), rgb)
#    image.save(image_path)  # Save the image to a specified path
#    return image

def create_color_image(r,g,b, image_path, width=100, height=100):
    #hex_code = hex_code.lstrip('#')
    #rgb = tuple(int(hex_code[i:i+2], 16) for i in (0, 2, 4))
    rgb = (r, g, b)
    image = Image.new('RGB', (width, height), rgb)
    image.save(image_path)
    return image

def hex_to_rgb(hex_code):
    hex_code = hex_code.lstrip('#')
    return tuple(int(hex_code[i:i+2], 16) for i in (0, 2, 4))

def lab_to_hex(Li, Ai, Bi):
    lab_color = LabColor(Li, Ai, Bi)
    rgb_color = convert_color(lab_color, sRGBColor)
    hex_code = rgb_color.get_rgb_hex()
    return hex_code

def _get_lab_color1(color):
    if not color.__class__.__name__ == 'LabColor':
        raise ValueError("Delta E functions can only be used with two LabColor objects.")
    return np.array([color.lab_l, color.lab_a, color.lab_b])

def _get_lab_color2(color):
    if not color.__class__.__name__ == 'LabColor':
        raise ValueError("Delta E functions can only be used with two LabColor objects.")
    return np.array([(color.lab_l, color.lab_a, color.lab_b)])

def delta_e_cie1976(color1, color2):
    color1 = _get_lab_color1(color1)
    color2 = _get_lab_color2(color2)
    delta_e = color_diff_matrix.delta_e_cie1976(color1, color2)
    return np.float64(delta_e)

def color(L,A,B):
    color = LabColor(lab_l=L, lab_a=A, lab_b=B)
    return color

def lab_to_rgb(l, a, b):
    lab_color = LabColor(lab_l=l, lab_a=a, lab_b=b)
    rgb_color = convert_color(lab_color, sRGBColor)
    r = float(rgb_color.rgb_r * 255)
    g = float(rgb_color.rgb_g * 255)
    b = float(rgb_color.rgb_b * 255)
    return r, g, b


def create_abs_coeff(total_thickness, R, G, B):

    if R == 0 or G == 0 or B == 0:
        return 0

    else:
        R_1 = R / 255
        G_1 = G / 255
        B_1 = B / 255

        R_tr = float(1 - R_1)
        Max_R = float(R_tr / R_1)
        Max_G = Max_R * G_1
        Max_B = Max_R * B_1

        Transmittance = (0.2125 * R_tr) + (0.7154 * Max_G) + (0.0721 * Max_B)

        if Transmittance <= 0:
            Transmittance = -(Transmittance)

        Absorbance = 2 - math.log10(Transmittance * 100)

        Abs_coeff = (2.303 * Absorbance) / total_thickness
        
        return Abs_coeff
