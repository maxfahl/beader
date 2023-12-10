# Beader

Beader is a Python application that creates a bead pattern from an image. It uses color analysis and image processing techniques to generate a pattern that can be used for bead art, pixel art, or similar crafts.

## Features

- Converts an image into a bead pattern.
- Allows customization of the bead pattern's rows, columns, and cell size.
- Supports background color customization.
- Detects a specified number of unique colors in the image.
- Optional contour color setting.
- Resizes the image before processing.
- Allows optional pixel sampling for color analysis.

## Installation

1. Clone the repository to your local machine.
2. Install the required Python packages listed in [`requirements.txt`]("requirements.txt") using pip:

```sh
pip install -r requirements.txt
```

## Usage

Run the [`main.py`]("main.py") script with the required arguments:

```sh
python main.py --image-path <path> --rows <rows> --columns <columns> --cell-size <cell_size> --background-color <background_color> --num-colors <num_colors> --contour-color <contour_color> --resize-width <resize_width> --resize-height <resize_height> --sample-size <sample_size>
```

| Argument              | Description                                                                 |
|-----------------------|-----------------------------------------------------------------------------|
| `--image-path`        | The path to the image to be processed.                                      |
| `--rows`              | Number of rows in the bead pattern                                          |
| `--columns`           | Number of columns in the bead pattern                                       |
| `--cell-size`         | Optional: Size of each cell in the bead pattern                             |
| `--background-color`  | Optional: Background color of the bead pattern                              |
| `--num-colors`        | Optional: Number of unique colors to detect in the image                    |
| `--contour-color`     | Optional: Color of the contour lines in the bead pattern                    |
| `--resize-width`      | Optional: Width to resize the image to before processing                    |
| `--resize-height`     | Optional: Height to resize the image to before processing                   |
| `--sample-size`       | Optional: Size of the sample to take from the image for color analysis      |
| `--ignore-colors`     | Optional: Colors to ignore in the bead pattern, e.g., 'black,white' or '#000000,#FFFFFF' |

## Dependencies

Beader uses the following Python packages:

- numpy
- Pillow
- argparse
- opencv-python
- colormath
- colour-science
- scikit-learn

## Contributing

Contributions are welcome. Please open an issue to discuss your ideas before making a pull request.

## License

This project is open source, under the terms of the [MIT license](https://opensource.org/licenses/MIT).