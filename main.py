from datetime import datetime
from pickletools import float8
import numpy as np
from PIL import Image, ImageColor, ImageDraw, ImageFilter, ImageFont
import argparse
import os
import mimetypes
import cv2
from PIL.Image import Resampling
from numpy import median
from collections import Counter
import colour
from sklearn.cluster import KMeans

font_size = 8  # Adjust the size to fit your cells
try:
    font = ImageFont.truetype("arial.ttf", font_size)
except IOError:
    font = ImageFont.load_default()


def delta_e_cie76(lab1, lab2):
    """
    Calculates the color difference between two LAB color values using the CIE76 formula.

    Parameters:
        lab1 (ndarray): The first LAB color value.
        lab2 (ndarray): The second LAB color value.

    Returns:
        float: The color difference between the two LAB color values.
    """
    return np.sqrt(np.sum((lab1 - lab2) ** 2, axis=-1))


def rgb_to_lab(color):
    """
    Convert an RGB color to LAB color space.

    Parameters:
    color (tuple): A tuple representing the RGB color values.

    Returns:
    tuple: A tuple representing the LAB color values.
    """
    rgb = colour.sRGB_to_XYZ(color)
    lab = colour.XYZ_to_Lab(rgb)
    return lab


def resize_image(image, resize_ratio):
    """
    Resize image for faster processing while maintaining aspect ratio.
    """
    width, height = image.size
    new_width = int(width * resize_ratio)
    new_height = int(height * resize_ratio)
    return image.resize((new_width, new_height), Resampling.LANCZOS)

def sample_pixels(image_array, sample_size):
    """
    Randomly sample pixels from the image.
    """
    print("Sampling pixels...")
    if sample_size is not None and sample_size < image_array.shape[0]:
        np.random.seed(0)  # For reproducibility
        indices = np.random.choice(image_array.shape[0], sample_size, replace=False)
        image_array = image_array[indices]

    return image_array.reshape((-1, 3))

def get_representative_colors(image_array, number_of_colors, ignore_colors):
    """
    Use k-means clustering to find most representative colors.
    """
    print("Finding representative colors...")
    ignore_labs = [rgb_to_lab(color) for color in ignore_colors] if ignore_colors else []
    color_difference_threshold = 10.0  # Adjust this value as needed


    while True:
        kmeans = KMeans(n_clusters=number_of_colors, n_init=10)
        kmeans.fit(image_array)
        centers = kmeans.cluster_centers_

        # Convert to integers and tuples
        centers = [tuple(int(value) for value in center) for center in centers]

        # Remove colors that are too close to contour_color or background_color
        centers = [color for color in centers if all(delta_e_cie76(rgb_to_lab(color), ignore_lab) > color_difference_threshold for ignore_lab in ignore_labs)]

        if len(centers) >= number_of_colors:
            break

    # Log the representative colors found
    print(f"  Done: {centers}")

    return centers

def find_representative_colors(image, number_of_colors, resize_ratio, sample_size=None, ignore_colors=None):
    """
    This function takes an image, number of representative colors to find, resize width and height,
    and optionally a sample size. The function resizes the image, randomly samples pixels from the
    resized image (if a sample size is provided), and then uses k-means clustering to find the
    representative colors in the sampled pixels.

    The function returns a list of tuples, where each tuple represents
    a color in RGB format and each value in the tuple is an integer
    between 0 and 255 inclusive.
    """
    print("Resizing the image...")
    resized_image = resize_image(image, resize_ratio)
    sampled_pixels = sample_pixels(np.array(resized_image), sample_size)
    return get_representative_colors(sampled_pixels, number_of_colors, ignore_colors)


# And then modify the closest_color function to use this simpler delta E calculation
def closest_color(target_color, colors_list):
    """
    Finds the closest color in a given list to a target color.

    Parameters:
    target_color (tuple): The target color in RGB format.
    colors_list (list): A list of colors in RGB format.

    Returns:
    tuple: The closest color in the list to the target color.
    """
    target_lab = rgb_to_lab(target_color)
    colors_lab = np.array([rgb_to_lab(color) for color in colors_list])
    distances = np.array([delta_e_cie76(target_lab, color_lab) for color_lab in colors_lab])
    index = np.argmin(distances)
    return tuple(colors_list[index])


def find_most_common_colors(image, num_colors):
    """
    Find the most common colors in an image.

    Args:
        image (PIL.Image.Image): The input image.
        num_colors (int): The number of most common colors to find.

    Returns:
        list: A list of RGB tuples representing the most common colors.
    """
    # Resize for faster processing, if the image is large
    image.thumbnail((200, 200), Resampling.LANCZOS)
    # Get colors from image and count them
    colors = image.getdata()
    color_counter = Counter(colors)
    # Find the most common colors
    most_common_colors = color_counter.most_common(num_colors)
    # Extract the color values
    most_common_colors = [color[0] for color in most_common_colors]

    return most_common_colors


def create_bead_pattern(original_image, rows, columns, cell_size, contour_color, background_color, limit_colors):
    """
    Create a bead pattern image based on the original image.

    Args:
        original_image (PIL.Image.Image): The original image to create the pattern from.
        rows (int): The number of rows in the pattern grid.
        columns (int): The number of columns in the pattern grid.
        cell_size (int): The size of each cell in the pattern grid.
        contour_color (tuple or None): The color of the contour lines. If None, no contour lines will be drawn.
        background_color (tuple): The background color of the pattern image.
        limit_colors (list): A list of colors to limit the pattern to.

    Returns:
        PIL.Image.Image: The bead pattern image.
    """
    # Resize the image to the desired grid size before edge detection
    resized_image = original_image.resize((columns, rows), Resampling.BILINEAR)
    resized_image_array = np.array(resized_image)

    # Convert the resized image to grayscale
    gray_image = resized_image.convert('L')
    gray_array = np.array(gray_image)

    edges = cv2.Canny(gray_array, threshold1=3, threshold2=50)

    # Create a new image for the pattern with the original dimensions
    pattern_image = Image.new('RGB', (columns * cell_size, rows * cell_size), background_color)
    draw = ImageDraw.Draw(pattern_image)

    # Draw the grid pattern
    for y in range(rows):
        for x in range(columns):
            # Get the color for each resized cell
            cell = resized_image_array[y:y + 1, x:x + 1].reshape(-1, 3)

            # Use k-means clustering to find the most representative color
            kmeans = KMeans(n_clusters=1, n_init=10)
            kmeans.fit(cell)
            representative_color = tuple(int(value) for value in kmeans.cluster_centers_[0])

            # Choose the closest color from the limit_colors
            closest_match = closest_color(representative_color, limit_colors)

            if contour_color:
                fill_color = contour_color if edges[y, x] != 0 else closest_match
            else:
                fill_color = closest_match

            # Convert fill_color to string representation of RGB color
            fill_color_str = f"rgb{fill_color}"

            # Draw the cell with a 1px border
            draw.rectangle(
                (float(x * cell_size + 1), float(y * cell_size + 1), float((x + 1) * cell_size - 2), float((y + 1) * cell_size - 2)),
                fill=fill_color_str
            )

    return pattern_image


def main():
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Create a bead pattern from an image.")
    parser.add_argument('--image-path', type=str, required=True, help="Path to the input image file")
    parser.add_argument('--rows', type=int, required=True, help="Number of rows in the bead pattern")
    parser.add_argument('--columns', type=int, required=True, help="Number of columns in the bead pattern")
    parser.add_argument('--cell-size', type=int, default=40, help="Size of each cell in the bead pattern")
    parser.add_argument('--background-color', type=str, default="#FFFFFF",
                        help="Background color of the bead pattern, e.g., 'white' or '#FFFFFF'")
    parser.add_argument('--num-colors', type=int, default=5,
                        help="Number of unique colors to detect in the image. Default to 3 if not set.")
    parser.add_argument('--contour-color', type=str, default=None,  # Set default to None
                        help="Optional: Color of the contour in the bead pattern, e.g., 'black' or '#000000'")
    parser.add_argument('--resize-ratio', type=float, default=1,
                        help="Ratio to resize the image by before processing. This can drastically increase the speed of processing. Default is 1 if not set.")
    parser.add_argument('--resize-height', type=int, default=200,
                        help="Height to resize the image to before processing, default is 200.")
    parser.add_argument('--sample-size', type=int, default=None,
                        help="Optional number of pixels to sample for color analysis. If not set, use all pixels.")
    # In the argument parsing section of the main function
    parser.add_argument('--ignore-colors', type=str, default=None,
                    help="Optional: Colors to ignore in the bead pattern, e.g., 'black,white' or '#000000,#FFFFFF'")

    # Parse arguments
    args = parser.parse_args()

    # Convert color arguments from string to RGB tuple
    contour_color = ImageColor.getrgb(args.contour_color) if args.contour_color else None
    background_color = ImageColor.getrgb(args.background_color)
    ignore_colors = [ImageColor.getrgb(color) for color in args.ignore_colors.split(',')] if args.ignore_colors else []
    ignore_colors.append(background_color)
    if contour_color:
        ignore_colors.append(contour_color)


    print("Loading the image...")
    # Load the image specified by the user
    original_image = Image.open(args.image_path).convert('RGB')

    # Find the representative colors
    limit_colors = find_representative_colors(
        original_image,
        args.num_colors,
        args.resize_ratio,
        args.sample_size,
        ignore_colors
    )

    print("Creating the bead pattern...")
    # Create the bead pattern
    pattern_image = create_bead_pattern(
        original_image,
        args.rows,
        args.columns,
        args.cell_size,
        contour_color,
        background_color,
        limit_colors
    )

    print("Saving the pattern image...")
    # Save the pattern image
    current_date = datetime.now().strftime('%y%m%d%-H%M')
    input_file_name = os.path.basename(args.image_path).split('.')[0]
    output_filename = f"image/out/{input_file_name}_{current_date}.png"
    pattern_image.save(output_filename)
    print(f"Pattern saved as {output_filename}")

    # Show the image
    pattern_image.show()

    # Print a message indicating the image that has been processed
    print(f"Image processed: {args.image_path}")


if __name__ == "__main__":
    main()