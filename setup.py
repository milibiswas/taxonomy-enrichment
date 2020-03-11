#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =============================================================================
# 
#       Name : setup.py
#       Description: To install the software
#       Created by : Mili Biswas
#       Created on: 21.02.2020
#
#       Dependency : requirements.txt file as well as python 3 environment
#
# =============================================================================

import subprocess
import sys

try:
    assert sys.version_info >= (3,)
except Exception:
    sys.exit("Error: Python version should be >= 3.x")

subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "./requirements.txt"])

