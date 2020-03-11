#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 10:17:08 2020

@author: milibiswas
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), './src/code/')))

import preprocess
import taxonomy_algorithm
import evaluation
import hypertree