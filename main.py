from datetime import datetime
import numpy as np
from PIL import Image, ImageColor, ImageDraw, ImageFilter, ImageFont
import argparse
import os
import mimetypes
import cv2
from PIL.Image import Resampling
from numpy import median
from sklearn.neighbors import NearestNeighbors

font_size = 8  # Adjust the size to fit your cells
try:
    font = ImageFont.truetype("arial.ttf", font_size)
except IOError:
    font = ImageFont.load_default()


def rgb_to_lab(color):
    # Convert a single RGB color to LAB
    return cv2.cvtColor(np.uint8([[color]]), cv2.COLOR_RGB2LAB)[0][0]


def closest_color(target_color, colors_list):
    # Convert the target and list colors to L*a*b* space
    target_lab = rgb_to_lab(target_color)
    colors_lab = np.array([rgb_to_lab(list(color)) for color in colors_list])

    # Use nearest neighbors in LAB space to find the closest color
    neighbors = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(colors_lab)
    distance, index = neighbors.kneighbors([target_lab])
    return tuple(colors_list[index[0][0]])


def create_bead_pattern(image_path, rows, columns, cell_size, contour_color, background_color, limit_colors):
    # Load and convert the image
    original_image = Image.open(image_path).convert('RGB')

    # Resize the image to the desired grid size before edge detection
    resized_image = original_image.resize((columns, rows), Resampling.BOX)
    resized_image_array = np.array(resized_image)

    # Convert the resized image to grayscale and apply a blur filter to reduce noise
    gray_image = resized_image.convert('L').filter(ImageFilter.MedianFilter(size=3))
    gray_array = np.array(gray_image)
    edges = cv2.Canny(gray_array, threshold1=50, threshold2=150)  # Adjusted thresholds

    # Create a new image for the pattern with the original dimensions
    pattern_image = Image.new('RGB', (columns * cell_size, rows * cell_size), background_color)
    draw = ImageDraw.Draw(pattern_image)

    # Draw the grid pattern
    for y in range(rows):
        for x in range(columns):
            # Get the median color for each resized cell
            cell = resized_image_array[y:y + 1, x:x + 1].reshape(-1, 3)
            median_color = tuple(int(v) for v in median(cell, axis=0))

            # Choose the closest color from the limit_colors
            closest_match = closest_color(median_color, limit_colors)

            if contour_color:
                fill_color = contour_color if edges[y, x] != 0 else closest_match
            else:
                fill_color = closest_match

            # Draw the cell
            draw.rectangle(
                [x * cell_size, y * cell_size, (x + 1) * cell_size - 1, (y + 1) * cell_size - 1],
                fill=fill_color,
                # outline=contour_color if contour_color else ImageColor.getrgb('#000000'),
                # width=1
            )

            # Position for the text will be at the top-left corner of the cell
            text_position = (x * cell_size + 5, y * cell_size + 3)  # +2 for a small margin

            # Text to indicate COL:ROW
            cell_text = f"{x}"

            # Draw the text with a thin black shadow
            shadow_position = (text_position[0] + 1, text_position[1] + 1)
            draw.text(shadow_position, cell_text, font=font, fill="black")
            draw.text(text_position, cell_text, font=font, fill="white")

    return pattern_image


def main():
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Create a bead pattern from an image.")
    # parser.add_argument('--image_path', type=str, required=True, help="Path to the input image file")
    parser.add_argument('--rows', type=int, required=True, help="Number of rows in the bead pattern")
    parser.add_argument('--columns', type=int, required=True, help="Number of columns in the bead pattern")
    parser.add_argument('--cell_size', type=int, required=True, help="Size of each cell in the bead pattern")
    parser.add_argument('--background_color', type=str, required=True,
                        help="Background color of the bead pattern, e.g., 'white' or '#FFFFFF'")
    parser.add_argument('--limit_colors', type=str, required=True,
                        help="Comma-separated list of colors to use in the bead pattern, e.g., 'black,white'")
    parser.add_argument('--contour_color', type=str, default=None,  # Set default to None
                        help="Optional: Color of the contour in the bead pattern, e.g., 'black' or '#000000'")

    # Parse arguments
    args = parser.parse_args()

    # Convert color arguments from string to RGB tuple
    contour_color = ImageColor.getrgb(args.contour_color) if args.contour_color else None
    background_color = ImageColor.getrgb(args.background_color)
    limit_colors = [ImageColor.getrgb(color) for color in args.limit_colors.split(',')]
    processed_images = []

    for image_path in os.listdir("image/in"):
        # Guess the type of the file based on its extension
        mime_type, _ = mimetypes.guess_type(image_path)

        # Check if the MIME type starts with 'image/'
        if mime_type and mime_type.startswith('image/'):
            full_path = f"image/in/{image_path}"
            pattern_image = create_bead_pattern(
                full_path,
                args.rows,
                args.columns,
                args.cell_size,
                contour_color,
                background_color,
                limit_colors
            )
            current_date = datetime.now().strftime('%y%m%d%-H%M')
            input_file_name = image_path.split('.')[0]
            output_filename = f"image/out/{input_file_name}_{current_date}.png"
            pattern_image.save(output_filename)
            processed_images.append(input_file_name)
            print(f"Pattern saved as {output_filename}")

            # show image
            pattern_image.show()

    # Print a message containing all images that has been processed
    print(f"Images processed: {','.join([img_path for img_path in processed_images])}")


if __name__ == "__main__":
    main()
