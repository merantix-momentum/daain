#!/usr/bin/env python
# -*- coding: utf-8 -*-

import ast
import itertools
import os
import re
import sys

from setuptools import find_packages, setup

SOURCE_DIR = "mxlabs_ood_detection"

if sys.version_info < (3, 8):
    sys.stderr.write("ERROR:  requires at least Python Version 3.6\n")
    sys.exit(1)

# Read package information from other files so that just one version has to be maintained.
_version_re = re.compile(r"__version__\s+=\s+(.*)")
with open("src/%s/__init__.py" % SOURCE_DIR, "rb") as f:
    init_contents = f.read().decode("utf-8")

    def get_var(var_name: str) -> str:
        """Parsing of mxlabs_ood_detection project infos defined in __init__.py"""
        pattern = re.compile(r"%s\s+=\s+(.*)" % var_name)
        match = pattern.search(init_contents).group(1)
        return str(ast.literal_eval(match))

    version = get_var("__version__")

# add tag to version if provided
if "--version_tag" in sys.argv:
    v_idx = sys.argv.index("--version_tag")
    version = version + "." + sys.argv[v_idx + 1]
    sys.argv.remove("--version_tag")
    sys.argv.pop(v_idx)

if os.path.exists("README.md"):
    with open("README.md") as fh:
        readme = fh.read()
else:
    readme = ""
if os.path.exists("HISTORY.md"):
    with open("HISTORY.md") as fh:
        history = fh.read().replace(".. :changelog:", "")
else:
    history = ""


def parse_req(spec: str) -> str:
    """ Parse package name==version out of requirments file"""
    if ";" in spec:
        # remove restriction
        spec, _ = [x.strip() for x in spec.split(";", 1)]
    if "#" in spec:
        # remove comment
        spec = spec.strip().split("#")[0]
    if "\\" in spec:
        # remove line breaks
        spec = spec.strip().split("\\")[0]
    if "--hash=" in spec:
        # remove line breaks
        spec = spec.strip().split("--hash=")[0]
    return spec


if os.path.exists("requirements.in"):
    with open("requirements.in") as fh:
        requirements = [parse_req(r) for r in fh.read().replace("\\\n", " ").split("\n") if parse_req(r) != ""]
else:
    requirements = []

# generate extras based on requirements files
extras_require = dict()
for a_extra in ["test"]:
    req_file = f"requirements.{a_extra}.in"
    if os.path.exists(req_file):
        with open(req_file) as fh:
            extras_require[a_extra] = [r for r in fh.read().split("\n") if ";" not in r]
    else:
        extras_require[a_extra] = []
extras_require["all"] = list(itertools.chain.from_iterable(extras_require.values()))

if os.path.exists("scripts"):
    SCRIPTS = [os.path.join("scripts", a) for a in os.listdir("scripts")]
else:
    SCRIPTS = []

cmdclass = dict()
# try to import sphinx
try:
    from sphinx.setup_command import BuildDoc

    cmdclass["build_sphinx"] = BuildDoc
except ImportError:
    sys.stdout.write("WARNING: sphinx not available, not building docs")

# Setup package using PIP
if __name__ == "__main__":
    setup(
        name=f"mxlabs-{SOURCE_DIR}",
        version=version,
        description="DAAIN: Detection of Adversarial and Anomalous Inputs using Normalising flows",
        long_description=f"{readme}\n\n{history}",
        author="Merantix AG",
        license="Proprietary",
        package_dir={"": "src"},
        packages=find_packages(where="./src"),
        scripts=SCRIPTS,
        include_package_data=True,
        install_requires=requirements,
        tests_require=extras_require["test"],
        extras_require=extras_require,
        cmdclass=cmdclass,
        classifiers=["Private :: Do Not Upload to pypi server"],
    )
