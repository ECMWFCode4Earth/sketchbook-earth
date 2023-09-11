# How to Execute the Notebooks

Choose from three methods to run the notebooks:

1. Binder
2. Google Colab
3. Locally on your machine

### Binder
Binder is the most straightforward method. Each notebook has a dedicated Binder button. Clicking this button will launch the corresponding Binder environment, handling all setup tasks for you.

### Google Colab
Each notebook features a dedicated Google Colab button. However, it won't automatically set up the entire environment and you'll need to install any missing Python modules. After accessing Google Colab, you can install missing modules using the following in a Colab cell:

```bash
!pip install <module>
```

For example, the following modules are not available in Google Colab but can be installed via:

```{dropdown} cdsapi
!pip install cdsapi
```

```{dropdown} cdsapi
See this [cartopy issue](https://github.com/SciTools/cartopy/issues/1490) for more info.
```bash
!pip install shapely cartopy --no-binary shapely --no-binary cartopy --force
```
```

```{dropdown} xeofs
!pip install xeofs
```

```{dropdown} tqdm
!pip install tqdm
```


### Local Machine
To run the notebooks on your computer:

**1. Clone the repository**

```bash
git clone https://github.com/ECMWFCode4Earth/sketchbook-earth.git
```

** 2. Navigate to the Repository and Install the Python Modules**

This is best done using `conda`:

```bash
cd sketchbook-earth
conda env create -f environment.yml
```

With this done, you're set to explore the local repository and run the notebooks on your computer as you would with any standard Jupyter notebook.