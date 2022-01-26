#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
    19/10/2021
        try to make extra_data for a run, test on a few hours
@author: jesse
"""

from utilities import fio

mr = "KI_run2"
for hour in range (4,18):
    fio.extra_data_make(mr,hour=hour)
