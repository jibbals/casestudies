#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
    19/10/2021
        try to make extra_data for a run, test on a few hours
@author: jesse
"""




from utilities import fio

mr = "badja_run3"
for hour in range (3,5):
    fio.read_extra_data(mr,hour=hour)
