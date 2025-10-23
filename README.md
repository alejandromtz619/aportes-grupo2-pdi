# Image enhancement using openCV

An image enhancement module used for improving the quality of images using different filters in openCV. 
It works better with gray-scale images. 

The filters used are median_filter for removing noise from the image, The histogram equalizer is used for contrast adjustment of the image and Gamma correction is also applied for preventing the image from darkening.

## Table of contents

- [Getting started](#getting-started)
- [Features](#features)
- [Usage](#usage)
- [Results](#results)
- [Want to Contribute?](#want-to-contribute)
- [Need Help / Support?](#need-help)
- [Collection of Other Components](#collection-of-components)
- [Changelog](#changelog)
- [Credits](#credits)
- [License](#license)
- [Keywords](#Keywords)

## Getting started

Prerequisites for running the code are:

Python >=3.8  
opencv-python >=4.5  
scipy >=1.8  
numpy >=1.20  
streamlit >=1.38 *(solo si se usa la interfaz web)*

Instalación rápida:

```bash
pip install opencv-python scipy numpy streamlit
```

## Features

- Pipeline modular que realza luminancia preservando el color en espacio LAB.
- Filtro mediano + bilateral (configurable) para reducir ruido sin perder bordes.
- CLAHE y corrección gamma personalizable para mejorar contraste y brillo.
- Máscara de realce (unsharp) opcional para recuperar nitidez.
- Interfaz CLI configurable y aplicación web (Streamlit) para pruebas rápidas.

## Usage

### CLI

Dentro del directorio del proyecto ejecuta:

```bash
python image_enhancement.py --input Dataset --output Results
```

Argumentos más útiles:

- `--gamma 1.2` ajusta el brillo global.
- `--no-color` fuerza salida en escala de grises.
- `--no-sharpen` desactiva el realce de nitidez.
- `--log-level DEBUG` muestra más información del proceso.

El script acepta un archivo individual (`--input path/a/imagen.jpg`) o una carpeta completa.

### Interfaz web

Si prefieres previsualizar resultados, lanza:

```bash
streamlit run streamlit_app.py
```

Esto abrirá una interfaz en el navegador donde puedes subir imágenes, ajustar parámetros con sliders y descargar la versión mejorada.

### Results
### Original Image
<img src="images/image1.jpg" width = "300" height = "225"/> <img src="images/image2.jpg" width = "300" height = "225"/>

### Processed Image
<img src="images/result1.jpg" width = "300" height = "225"/> <img src="images/result2.jpg" width = "300" height = "225"/>

## Want to Contribute?

- Created something awesome, made this code better, added some functionality, or whatever (this is the hardest part).
- [Fork it](http://help.github.com/forking/).
- Create new branch to contribute your changes.
- Commit all your changes to your branch.
- Submit a [pull request](http://help.github.com/pull-requests/).

-----

## Need Help? 

We also provide a free, basic support for all users who want to use image processing techniques for their projects. In case you want to customize this image enhancement technique for your development needs, then feel free to contact our [AI/ML developers](https://www.weblineindia.com/ai-ml-dl-development.html).

-----

## Collection of Components

We have built many other components and free resources for software development in various programming languages. Kindly click here to view our [Free Resources for Software Development](https://www.weblineindia.com/communities.html).

------

## Changelog

Detailed changes for each release are documented in [CHANGELOG.md](./CHANGELOG.md).

## Credits

Refered OpenCV image processing and filtering techniques.  [opencv](https://docs.opencv.org/3.4/index.html).

## License

[MIT](LICENSE)

[mit]: https://github.com/miguelmota/is-valid-domain/blob/e48e90f3ecd55431bbdba950eea013c2072d2fac/LICENSE

## Keywords

 image-processing, image-filters, image-enhancement-opencv,opencv-image-processing,image-denoising,histogram-equalizer
