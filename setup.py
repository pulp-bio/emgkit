"""Setup script.


Copyright 2023 Mattia Orlandi

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from setuptools import setup

setup(
    name="emgkit",
    version="1.0.0",
    description="A toolkit for the analysis and processing of EMG signals",
    author="Mattia Orlandi",
    author_email="mattia.orlandi@unibo.it",
    packages=[
        "emgkit",
        "emgkit.decomposition",
        "emgkit.features",
        "emgkit.plotting",
        "emgkit.preprocessing",
        "emgkit.spike_stats",
        "emgkit.utils",
    ],
)
