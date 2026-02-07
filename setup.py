"""CMake-based build for pybind11 extension module."""

import os
import subprocess
import sys
from pathlib import Path

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=""):
        super().__init__(name, sources=[])
        self.sourcedir = os.fspath(Path(sourcedir).resolve())


class CMakeBuild(build_ext):
    def build_extension(self, ext):
        extdir = Path(self.get_ext_fullpath(ext.name)).resolve().parent
        cfg = "Release"

        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir / 'cphnsw'}",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
            f"-DCMAKE_BUILD_TYPE={cfg}",
            "-DCPHNSW_BUILD_PYTHON=ON",
            "-DCPHNSW_BUILD_TESTS=OFF",
            "-DCPHNSW_BUILD_EVAL=OFF",
        ]

        build_args = ["--config", cfg, "--target", "_core"]

        build_temp = Path(self.build_temp) / ext.name
        build_temp.mkdir(parents=True, exist_ok=True)

        subprocess.run(
            ["cmake", ext.sourcedir, *cmake_args],
            cwd=build_temp, check=True,
        )
        subprocess.run(
            ["cmake", "--build", ".", *build_args],
            cwd=build_temp, check=True,
        )


setup(
    ext_modules=[CMakeExtension("cphnsw._core")],
    cmdclass={"build_ext": CMakeBuild},
)
