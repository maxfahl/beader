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
2. Install the required Python packages listed in [`requirements.txt`](command:_github.copilot.openRelativePath?%5B%22requirements.txt%22%5D "requirements.txt") using pip:

```sh
pip install -r requirements.txt
```

## Usage

Run the [`main.py`](command:_github.copilot.openRelativePath?%5B%22main.py%22%5D "main.py") script with the required arguments:

```sh
python main.py --rows <rows> --columns <columns> --cell_size <cell_size> --background_color <background_color> --num-colors <num_colors> --contour_color <contour_color> --resize-width <resize_width> --resize-height <resize_height> --sample-size <sample_size>
```

| Argument          | Description                                                                 |
|-------------------|-----------------------------------------------------------------------------|
| `--rows`          | Number of rows in the bead pattern                                          |
| `--columns`       | Number of columns in the bead pattern                                       |
| `--cell_size`     | Size of each cell in the bead pattern                                       |
| `--background_color` | Background color of the bead pattern                                     |
| `--num_colors`    | Number of unique colors to detect in the image                              |
| `--contour_color` | Color of the contour lines in the bead pattern                              |
| `--resize_width`  | Width to resize the image to before processing                              |
| `--resize_height` | Height to resize the image to before processing                             |
| `--sample_size`   | Size of the sample to take from the image for color analysis (optional)     |

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