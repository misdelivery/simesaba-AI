from distutils.core import setup
import os
import sys
import streamlit as st

def package_install():
  sys.path.append('/home/appuser/.local/bin')
  os.system("pip install Cython numpy")

package_install()

from Cython.Build import cythonize
import numpy

setup(
  name = 'monotonic_align',
  ext_modules = cythonize("core.pyx"),
  include_dirs=[numpy.get_include()]
)
