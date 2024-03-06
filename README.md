# SHARPESST
Solar Heliophysics Algorithms for Rectification of PSFs and Estimation of Sources Using Sparse Transforms

This software was primarily developed for correction of PSF artifacts for solar orbiter SPICE, see https://ui.adsabs.harvard.edu/abs/2023A%26A...678A..52P/abstract. However, the components are relevant to wide variety of other applications that are still being developed. The initial package here is built around the PSF artifact correction application, and include an example notebook with accompanying data file.

1. Standard instructions using Conda:

A standard set of Python packages will be needed. These should include jupyter, numpy, numba, matplotlib, scipy, and astropy. I recommend using a Python packaging/environment manager -- I personally use mini-forge which is a compact free and open source implementation of Conda/Anaconda. It works on Windows, Mac, and Linux; installation links and instructions can be found here:

	https://github.com/conda-forge/miniforge/#download

After installing miniforge, to create a Python environment with the required Python packages, type the following commands:

	conda create -n sharpesst

 	conda activate sharpesst

 	conda install jupyter numpy numba matplotlib scipy astropy

It will probably take a while for it to determine all of the dependencies, but once it does, answer 'y' to finish the installation. You should then be good to start Jupyter in a command line terminal from the top directory of the repo:
	jupyter notebook --no-browser
	
Copy the link jupyter prints into your web browser and it will take you to the notebook. Once you're finished you can type 'conda deactivate' in the command line window to deactivate the sharpesst environment.

2. Instructions for using pip:

Similar to Conda, with pip it's recommended that you create a python virtual environment to encapsulate the packages needed for a given application. This helps minimizes conflicts when there are a large number of packages installed on a system (not an issue in the short term, but very much so in the long term). Refer to the Python documents here:
	https://docs.python.org/3/library/venv.html

Once the pip virtual environment is set up and active, you should be able to install the required packages with
	pip install jupyter numpy numba matplotlib scipy astropy

Then run the notebook the same way as with Conda.

3. Installing the PSF modules to system:

Installing the PSF modules themselves, whether within pip or a Conda environment, is not required. It is possible to simply point python's include command at the SHARPESST directory. This is what the example notebook does:
	sharpesst_path = os.path.join(base_path,'SHARPESST')
	path.append(sharpesst_path) 

Avoiding the install makes the whole thing more portable and also allows tweaking the code without doing a reinstall.

If doing an install of the PSF packages themselves is desired there is a pip install script included which will also installed the dependency packages (if using Conda, it's much better to install the dependencies manually as described above; see https://www.anaconda.com/blog/using-pip-in-a-conda-environment)

To run it, move to the SHARPESST (all caps) subdirectory/folder and type:

	pip install -e .

Note that the trailing period is part of the command!
